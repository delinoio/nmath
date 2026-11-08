//! Shared storage and strided array views.

use std::sync::{
    Arc, RwLock,
    atomic::{AtomicU64, Ordering},
};

use crate::{
    buffer::{Buffer, DenseArray, Scalar, checked_byte_len},
    dtype::DType,
    error::{Error, ErrorCode, Result},
    memory::{MemoryBudget, Reservation},
    shape::{
        Selector, advanced_axis, broadcast_shapes, broadcast_strides, contiguous_strides,
        for_each_offset, index_offsets_budgeted, index_shape, normalize_index, resolve_slice,
    },
    sync::{CancellationToken, FifoGuard, FifoLock, Ticket},
};

static NEXT_STORAGE_ID: AtomicU64 = AtomicU64::new(1);

/// Reference-counted storage owned by one context. Views keep it alive.
#[derive(Debug)]
pub struct Storage {
    id: u64,
    context_id: u64,
    data: RwLock<Buffer>,
    writes: FifoLock,
    _reservation: Reservation,
}

impl std::fmt::Debug for FifoLock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("FifoLock")
    }
}

/// A strided view of a [`Storage`]. Cloning shares the storage.
#[derive(Clone, Debug)]
pub struct NdArray {
    storage: Arc<Storage>,
    offset: isize,
    shape: Vec<usize>,
    strides: Vec<isize>,
}

impl NdArray {
    /// Wraps an owned dense array (already charged to its budget) as new
    /// storage.
    pub fn from_dense(context_id: u64, dense: DenseArray) -> NdArray {
        let strides = contiguous_strides(&dense.shape);
        NdArray {
            storage: Arc::new(Storage {
                id: NEXT_STORAGE_ID.fetch_add(1, Ordering::Relaxed),
                context_id,
                data: RwLock::new(dense.buffer),
                writes: FifoLock::new(),
                _reservation: dense.reservation,
            }),
            offset: 0,
            shape: dense.shape,
            strides,
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn dtype(&self) -> DType {
        self.storage.data.read().unwrap().dtype()
    }

    pub fn context_id(&self) -> u64 {
        self.storage.context_id
    }

    pub fn storage_id(&self) -> u64 {
        self.storage.id
    }

    pub fn shares_storage(&self, other: &NdArray) -> bool {
        Arc::ptr_eq(&self.storage, &other.storage)
    }

    /// Reads the visible elements into a new contiguous buffer.
    pub fn to_buffer(&self) -> Buffer {
        let data = self.storage.data.read().unwrap();
        let mut offsets = Vec::with_capacity(self.size());
        for_each_offset(&self.shape, &self.strides, self.offset, |o| offsets.push(o));
        data.gather(&offsets)
    }

    /// Copies the visible elements into a snapshot charged to `budget`.
    pub fn snapshot(&self, budget: &MemoryBudget) -> Result<DenseArray> {
        self.snapshot_cancellable(budget, &CancellationToken::new())
    }

    /// Materialize in logical C order with bounded cancellation and no
    /// element-sized offsets allocation. Reserve before allocating output.
    pub(crate) fn snapshot_cancellable(
        &self,
        budget: &MemoryBudget,
        token: &CancellationToken,
    ) -> Result<DenseArray> {
        token.check()?;
        let dtype = self.dtype();
        let size = self.size();
        let bytes = crate::buffer::checked_byte_len(size, dtype)?;
        let charge = budget.reserve(bytes as u64)?;
        let mut output = Buffer::filled(dtype, size, Scalar::Int(0))?;
        let source = self.storage.data.read().unwrap();
        for i in 0..size {
            if i % 1024 == 0 {
                token.check()?;
            }
            let mut linear = i;
            let mut offset = self.offset;
            for (&dim, &stride) in self.shape.iter().zip(&self.strides).rev() {
                offset += (linear % dim) as isize * stride;
                linear /= dim;
            }
            output.set_from(i, &source, offset as usize);
        }
        token.check()?;
        DenseArray::new(self.shape.clone(), output, charge)
    }

    /// A rank-0 element read.
    pub fn get_scalar(&self, index: &[i64]) -> Result<Scalar> {
        if index.len() != self.ndim() {
            return Err(Error::new(
                ErrorCode::IndexInvalid,
                format!("Expected {} indices, got {}.", self.ndim(), index.len()),
            ));
        }
        let mut offset = self.offset;
        for (axis, (&i, (&len, &stride))) in index
            .iter()
            .zip(self.shape.iter().zip(&self.strides))
            .enumerate()
        {
            offset += normalize_index(i, len, axis)? as isize * stride;
        }
        Ok(self.storage.data.read().unwrap().get(offset as usize))
    }

    /// Basic indexing returns a view that shares storage, as in NumPy.
    pub fn view(&self, selectors: &[Selector]) -> Result<NdArray> {
        if selectors.iter().any(|s| !s.is_basic()) {
            return Err(Error::new(
                ErrorCode::IndexInvalid,
                "Only integer and slice selectors create views; use a copy for array or mask \
                 selectors.",
            ));
        }
        index_shape(&self.shape, selectors)?;
        let mut offset = self.offset;
        let mut shape = Vec::new();
        let mut strides = Vec::new();
        for (axis, (&len, &stride)) in self.shape.iter().zip(&self.strides).enumerate() {
            match selectors.get(axis) {
                None => {
                    shape.push(len);
                    strides.push(stride);
                }
                Some(Selector::Index(i)) => {
                    offset += normalize_index(*i, len, axis)? as isize * stride;
                }
                Some(Selector::Slice { start, stop, step }) => {
                    let (s, st, n) = resolve_slice(*start, *stop, *step, len)?;
                    if n > 0 {
                        offset += s * stride;
                    }
                    shape.push(n);
                    // A stride is never applied on an axis with at most one
                    // element, and extreme steps such as `i64::MAX` would
                    // overflow the multiplication.
                    strides.push(if n <= 1 { 0 } else { st * stride });
                }
                Some(_) => unreachable!(),
            }
        }
        Ok(NdArray {
            storage: self.storage.clone(),
            offset,
            shape,
            strides,
        })
    }

    /// Permutes metadata while retaining the same storage and write ordering.
    pub(crate) fn transpose(&self, axes: &[usize]) -> NdArray {
        NdArray {
            storage: self.storage.clone(),
            offset: self.offset,
            shape: axes.iter().map(|&axis| self.shape[axis]).collect(),
            strides: axes.iter().map(|&axis| self.strides[axis]).collect(),
        }
    }

    pub(crate) fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// Returns a shared C-order reshape when the existing strides permit it.
    pub(crate) fn reshape_view(&self, shape: &[usize]) -> Option<NdArray> {
        let strides = crate::layout::reshape_strides(&self.shape, &self.strides, shape)?;
        Some(NdArray {
            storage: self.storage.clone(),
            offset: self.offset,
            shape: shape.to_vec(),
            strides,
        })
    }

    /// Copying (advanced) indexing into a new dense array charged to `budget`.
    pub fn take(&self, selectors: &[Selector], budget: &MemoryBudget) -> Result<DenseArray> {
        self.take_cancellable(selectors, budget, &CancellationToken::new())
    }

    pub(crate) fn take_cancellable(
        &self,
        selectors: &[Selector],
        budget: &MemoryBudget,
        token: &CancellationToken,
    ) -> Result<DenseArray> {
        token.check()?;
        let shape = index_shape(&self.shape, selectors)?;
        let size = shape
            .iter()
            .try_fold(1usize, |n, &dim| n.checked_mul(dim))
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::MemoryBudgetExceeded,
                    "Index result size exceeds the supported vector capacity.",
                )
            })?;
        let dtype = self.dtype();
        let bytes = crate::buffer::checked_byte_len(size, dtype)?;
        let reservation = budget.reserve(bytes as u64)?;
        let mask_count = if size == 0 {
            0
        } else {
            selectors
                .iter()
                .filter(|selector| matches!(selector, Selector::Mask(_)))
                .count()
        };
        let scratch_bytes = mask_count
            .checked_mul(std::mem::size_of::<crate::shape::MaskCursor>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .filter(|&bytes| bytes <= isize::MAX as u64)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::MemoryBudgetExceeded,
                    "Index metadata size exceeds the supported vector capacity.",
                )
            })?;
        let _scratch = budget.reserve(scratch_bytes)?;
        let mut mask_cursors = Vec::with_capacity(mask_count);
        if size != 0 {
            for selector in selectors {
                if let Selector::Mask(mask) = selector {
                    mask_cursors.push(crate::shape::MaskCursor::new(mask, token)?);
                }
            }
        }
        let mut buffer = Buffer::filled(dtype, size, Scalar::Int(0))?;
        let source = self.storage.data.read().unwrap();
        let output_advanced_axis = advanced_axis(selectors);
        let mut visited = 0usize;
        let mut write_output = |output_index: usize| -> Result<()> {
            if visited % 1024 == 0 {
                token.check()?;
            }
            let offset = crate::shape::indexed_offset(
                &self.shape,
                &self.strides,
                self.offset,
                selectors,
                &shape,
                output_index,
                &mut mask_cursors,
                token,
            )?;
            buffer.set_from(output_index, &source, offset);
            visited += 1;
            Ok(())
        };
        if size != 0 {
            if let Some(axis) = output_advanced_axis.filter(|&axis| axis > 0) {
                let advanced_len = shape[axis];
                let trailing = shape[axis + 1..].iter().product::<usize>();
                let leading = size / (advanced_len * trailing);
                for advanced_index in 0..advanced_len {
                    for leading_index in 0..leading {
                        let prefix = leading_index * advanced_len * trailing;
                        for trailing_index in 0..trailing {
                            write_output(prefix + advanced_index * trailing + trailing_index)?;
                        }
                    }
                }
            } else {
                for output_index in 0..size {
                    write_output(output_index)?;
                }
            }
        }
        token.check()?;
        DenseArray::new(shape, buffer, reservation)
    }

    /// Takes a place in this storage's write line. Call at request time.
    pub fn write_ticket(&self) -> Ticket {
        self.storage.writes.ticket()
    }

    /// Writes `value` into the selected elements. The write commits completely
    /// or not at all: everything that can fail runs before the storage is
    /// touched.
    pub fn assign(
        &self,
        _turn: &FifoGuard,
        selectors: &[Selector],
        value: &DenseArray,
        token: &CancellationToken,
    ) -> Result<()> {
        // Direct Rust writes have no context budget argument. Keep this API
        // usable with checked allocation sizes; Context::assign supplies its
        // actual CPU budget through assign_with_budget.
        self.assign_with_budget(
            _turn,
            selectors,
            value,
            token,
            &MemoryBudget::new(crate::memory::Device::Cpu, u64::MAX),
        )
    }

    pub(crate) fn assign_with_budget(
        &self,
        _turn: &FifoGuard,
        selectors: &[Selector],
        value: &DenseArray,
        token: &CancellationToken,
        budget: &MemoryBudget,
    ) -> Result<()> {
        let target_shape = index_shape(&self.shape, selectors)?;
        let out = broadcast_shapes(&value.shape, &target_shape)?;
        if out != target_shape {
            return Err(Error::new(
                ErrorCode::ShapeBroadcast,
                format!(
                    "A value of shape {:?} cannot be assigned to a selection of shape {:?}.",
                    value.shape, target_shape
                ),
            ));
        }
        let dtype = self.dtype();
        if matches!(value.dtype(), DType::Complex64 | DType::Complex128)
            && !matches!(dtype, DType::Complex64 | DType::Complex128 | DType::Bool)
        {
            return Err(Error::new(
                ErrorCode::DTypeUnsupported,
                "Complex writes to real storage require an explicit cast before assignment.",
            ));
        }
        if let (DType::Str(target), Buffer::Unicode { width, values }) = (dtype, &value.buffer) {
            if *width > target
                && values
                    .chunks_exact(*width)
                    .any(|row| row[target..].iter().any(|&c| c != 0))
            {
                return Err(Error::new(
                    ErrorCode::DTypeUnsupported,
                    "String assignment would truncate a value. Cast explicitly to the target \
                     width first.",
                ));
            }
        }
        if dtype.temporal_unit().is_some() && value.dtype() != dtype {
            return Err(Error::new(
                ErrorCode::DTypeUnsupported,
                "Temporal assignment requires matching kind and unit. Cast explicitly before \
                 assignment.",
            ));
        }
        let cast_bytes = checked_byte_len(value.size(), dtype)?;
        let _cast_charge = budget.reserve(cast_bytes as u64)?;
        let source = value.buffer.cast_cancellable(dtype, token)?;
        let offsets =
            index_offsets_budgeted(&self.shape, &self.strides, self.offset, selectors, budget)?;
        let rank_bytes = value
            .shape
            .len()
            .checked_add(
                target_shape
                    .len()
                    .checked_mul(2)
                    .ok_or_else(crate::shape::index_scratch_overflow)?,
            )
            .and_then(|n| n.checked_mul(std::mem::size_of::<usize>()))
            .ok_or_else(crate::shape::index_scratch_overflow)?;
        let source_bytes = offsets
            .values
            .len()
            .checked_mul(std::mem::size_of::<usize>())
            .and_then(|n| n.checked_add(rank_bytes))
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(crate::shape::index_scratch_overflow)?;
        let _source_charge = budget.reserve(source_bytes as u64)?;
        let strides = broadcast_strides(&value.shape, &target_shape);
        let mut source_offsets = Vec::with_capacity(offsets.values.len());
        for_each_offset(&target_shape, &strides, 0, |o| source_offsets.push(o));
        token.check()?;
        let _snapshot_gate = crate::table::SNAPSHOT_GATE.lock().unwrap();
        token.check()?;
        let mut data = self.storage.data.write().unwrap();
        for (&dst, &src) in offsets.values.iter().zip(&source_offsets) {
            data.set_from(dst, &source, src);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::Device;

    fn array(shape: Vec<usize>, values: Vec<f64>) -> NdArray {
        NdArray::from_dense(
            1,
            DenseArray::new(shape, Buffer::F64(values), Reservation::none()).unwrap(),
        )
    }

    fn slice(start: Option<i64>, stop: Option<i64>, step: Option<i64>) -> Selector {
        Selector::Slice { start, stop, step }
    }

    #[test]
    fn views_share_storage_and_see_writes() {
        let a = array(vec![2, 3], (0..6).map(|v| v as f64).collect());
        let row = a.view(&[Selector::Index(1)]).unwrap();
        assert_eq!(row.shape(), &[3]);
        assert!(row.shares_storage(&a));
        let token = CancellationToken::new();
        let turn = a.write_ticket().wait(&token).unwrap();
        let ten = DenseArray::new(vec![], Buffer::F64(vec![10.0]), Reservation::none()).unwrap();
        a.assign(
            &turn,
            &[Selector::Index(1), Selector::Index(0)],
            &ten,
            &token,
        )
        .unwrap();
        assert_eq!(row.to_buffer(), Buffer::F64(vec![10.0, 4.0, 5.0]));
    }

    #[test]
    fn reversed_and_stepped_views() {
        let a = array(vec![5], (0..5).map(|v| v as f64).collect());
        let rev = a.view(&[slice(None, None, Some(-2))]).unwrap();
        assert_eq!(rev.to_buffer(), Buffer::F64(vec![4.0, 2.0, 0.0]));
        assert_eq!(rev.get_scalar(&[-1]).unwrap(), Scalar::Float(0.0));
    }

    #[test]
    fn extreme_slice_steps_select_one_element() {
        let a = array(vec![3, 2], (0..6).map(|v| v as f64).collect());
        let first = a.view(&[slice(None, None, Some(i64::MAX))]).unwrap();
        assert_eq!(first.shape(), &[1, 2]);
        assert_eq!(first.to_buffer(), Buffer::F64(vec![0.0, 1.0]));
        let last = a.view(&[slice(None, None, Some(i64::MIN))]).unwrap();
        assert_eq!(last.to_buffer(), Buffer::F64(vec![4.0, 5.0]));
        let budget = MemoryBudget::new(Device::Cpu, u64::MAX);
        let taken = a
            .take(
                &[slice(None, None, Some(i64::MIN)), Selector::Index(0)],
                &budget,
            )
            .unwrap();
        assert_eq!(taken.buffer, Buffer::F64(vec![4.0]));
    }

    #[test]
    fn failed_assignment_preserves_target() {
        let a = array(vec![3], vec![1.0, 2.0, 3.0]);
        let token = CancellationToken::new();
        let turn = a.write_ticket().wait(&token).unwrap();
        let bad =
            DenseArray::new(vec![2], Buffer::F64(vec![9.0, 9.0]), Reservation::none()).unwrap();
        assert!(a.assign(&turn, &[], &bad, &token).is_err());
        token.cancel();
        let ok_shape =
            DenseArray::new(vec![3], Buffer::F64(vec![9.0; 3]), Reservation::none()).unwrap();
        assert_eq!(
            a.assign(&turn, &[], &ok_shape, &token).unwrap_err().code,
            ErrorCode::Cancelled
        );
        assert_eq!(a.to_buffer(), Buffer::F64(vec![1.0, 2.0, 3.0]));
    }

    #[test]
    fn take_charges_budget() {
        let a = array(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
        let budget = MemoryBudget::new(Device::Cpu, 16);
        let taken = a.take(&[Selector::Indices(vec![3, 0])], &budget).unwrap();
        assert_eq!(taken.buffer, Buffer::F64(vec![4.0, 1.0]));
        assert_eq!(budget.usage().used, 16);
        assert!(a.take(&[Selector::Indices(vec![0])], &budget).is_err());
    }

    #[test]
    fn copying_mask_after_retained_axis_preserves_c_order() {
        let a = array(vec![2, 5], (0..10).map(|v| v as f64).collect());
        let budget = MemoryBudget::new(Device::Cpu, 64);
        let taken = a
            .take(
                &[
                    slice(None, None, None),
                    Selector::Mask(vec![false, true, false, false, true]),
                ],
                &budget,
            )
            .unwrap();
        assert_eq!(taken.shape, vec![2, 2]);
        assert_eq!(taken.buffer, Buffer::F64(vec![1.0, 4.0, 6.0, 9.0]));
    }

    #[test]
    fn copying_index_preserves_separated_order_and_reversed_values() {
        let a = array(vec![2, 3, 4], (0..24).map(|v| v as f64).collect());
        let reversed = a
            .view(&[Selector::Slice {
                start: None,
                stop: None,
                step: Some(-1),
            }])
            .unwrap();
        let budget = MemoryBudget::new(Device::Cpu, 48);
        let selected = reversed
            .take(
                &[
                    Selector::Index(0),
                    Selector::Slice {
                        start: None,
                        stop: None,
                        step: Some(2),
                    },
                    Selector::Indices(vec![3, 0]),
                ],
                &budget,
            )
            .unwrap();
        assert_eq!(selected.shape, vec![2, 2]);
        assert_eq!(selected.buffer, Buffer::F64(vec![15.0, 23.0, 12.0, 20.0]));
        assert_eq!(budget.usage().used, 32);
    }

    #[test]
    fn copying_index_releases_reservation_on_failure_or_cancellation() {
        let a = array(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
        let budget = MemoryBudget::new(Device::Cpu, 8);
        let token = CancellationToken::new();
        assert_eq!(
            a.take_cancellable(&[Selector::Indices(vec![0, 1])], &budget, &token)
                .unwrap_err()
                .code,
            ErrorCode::MemoryBudgetExceeded
        );
        assert_eq!(budget.usage().used, 0);
        token.cancel();
        assert_eq!(
            a.take_cancellable(&[Selector::Indices(vec![0])], &budget, &token)
                .unwrap_err()
                .code,
            ErrorCode::Cancelled
        );
        assert_eq!(budget.usage().used, 0);
    }

    #[test]
    fn copying_mask_charges_cursor_metadata() {
        let a = array(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
        let scratch = std::mem::size_of::<crate::shape::MaskCursor>() as u64;
        let budget = MemoryBudget::new(Device::Cpu, 16 + scratch);
        let selected = a
            .take(&[Selector::Mask(vec![true, false, true, false])], &budget)
            .unwrap();
        assert_eq!(selected.buffer, Buffer::F64(vec![1.0, 3.0]));
        assert_eq!(budget.usage().peak, 16 + scratch);
        assert_eq!(budget.usage().used, 16);
    }

    #[test]
    fn empty_mask_needs_no_cursor_storage() {
        let a = array(vec![3], vec![1.0, 2.0, 3.0]);
        let budget = MemoryBudget::new(Device::Cpu, 0);
        let selected = a
            .take(&[Selector::Mask(vec![false, false, false])], &budget)
            .unwrap();
        assert_eq!(selected.shape, vec![0]);
        assert_eq!(selected.buffer, Buffer::F64(vec![]));
        assert_eq!(budget.usage().used, 0);
    }
}
