// SPDX-License-Identifier: Apache-2.0
//! Allocation-free validation before handing untrusted lengths to codecs.
use super::{Result, format_error};

/// Compact Thrift scanner. Container counts, nesting and binary lengths must
/// fit the input. This prevents length-driven allocation in the Parquet parser.
pub(super) struct Compact<'a> {
    bytes: &'a [u8],
    pub pos: usize,
    pub unpacked: u64,
    pub packed: u64,
    pub values: u64,
}
impl<'a> Compact<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            pos: 0,
            unpacked: 0,
            packed: 0,
            values: 0,
        }
    }

    fn byte(&mut self) -> Result<u8> {
        let v = *self.bytes.get(self.pos).ok_or_else(format_error)?;
        self.pos += 1;
        Ok(v)
    }

    fn skip(&mut self, n: u64) -> Result<()> {
        let n = usize::try_from(n).map_err(|_| format_error())?;
        self.pos = self
            .pos
            .checked_add(n)
            .filter(|&p| p <= self.bytes.len())
            .ok_or_else(format_error)?;
        Ok(())
    }

    fn uint(&mut self) -> Result<u64> {
        let mut out = 0;
        for shift in (0..70).step_by(7) {
            let b = self.byte()?;
            if shift == 63 && b > 1 {
                return Err(format_error());
            }
            out |= u64::from(b & 127) << shift;
            if b & 128 == 0 {
                return Ok(out);
            }
        }
        Err(format_error())
    }

    fn int(&mut self) -> Result<i64> {
        let n = self.uint()?;
        Ok(((n >> 1) as i64) ^ -((n & 1) as i64))
    }

    pub fn structure(&mut self, depth: usize, page: bool) -> Result<()> {
        if depth > 32 {
            return Err(format_error());
        }
        let mut id = 0i64;
        loop {
            let h = self.byte()?;
            if h == 0 {
                return Ok(());
            }
            id = if h >> 4 == 0 {
                self.int()?
            } else {
                id.checked_add(i64::from(h >> 4)).ok_or_else(format_error)?
            };
            if page && depth == 0 && matches!(id, 2 | 3) && h & 15 == 5 {
                let n = u64::try_from(self.int()?).map_err(|_| format_error())?;
                if id == 2 {
                    self.unpacked = n;
                } else {
                    self.packed = n;
                }
            } else if page && depth == 1 && id == 1 && h & 15 == 5 {
                self.values = self
                    .values
                    .max(u64::try_from(self.int()?).map_err(|_| format_error())?);
            } else if h & 15 == 12 {
                self.structure(depth + 1, page)?;
            } else {
                self.value(h & 15, depth + 1, true)?;
            }
        }
    }

    fn value(&mut self, kind: u8, depth: usize, field: bool) -> Result<()> {
        if depth > 32 {
            return Err(format_error());
        }
        match kind {
            1 | 2 => {
                if !field {
                    self.byte()?;
                }
            }
            3 => {
                self.byte()?;
            }
            4..=6 => {
                self.uint()?;
            }
            7 => self.skip(8)?,
            8 => {
                let n = self.uint()?;
                self.skip(n)?;
            }
            9 | 10 => {
                let h = self.byte()?;
                let n = if h >> 4 == 15 {
                    self.uint()?
                } else {
                    u64::from(h >> 4)
                };
                if n > (self.bytes.len() - self.pos) as u64 {
                    return Err(format_error());
                }
                for _ in 0..n {
                    self.value(h & 15, depth + 1, false)?;
                }
            }
            11 => {
                let n = self.uint()?;
                if n > (self.bytes.len() - self.pos) as u64 / 2 {
                    return Err(format_error());
                }
                if n > 0 {
                    let h = self.byte()?;
                    for _ in 0..n {
                        self.value(h >> 4, depth + 1, false)?;
                        self.value(h & 15, depth + 1, false)?;
                    }
                }
            }
            12 => self.structure(depth + 1, false)?,
            _ => return Err(format_error()),
        }
        Ok(())
    }
}
