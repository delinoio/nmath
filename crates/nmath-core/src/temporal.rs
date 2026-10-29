// SPDX-License-Identifier: Apache-2.0
//! Exact temporal unit conversion. Dates use the proleptic Gregorian calendar.
use crate::{
    dtype::{TimeBase, TimeUnit, unsupported},
    error::Result,
};
pub const NAT: i64 = i64::MIN;

fn ratio(value: i128, from: i128, to: i128) -> Result<i64> {
    let (mut a, mut b) = (from, to);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    let (num, den) = (from / a, to / a);
    // NumPy rejects a conversion whose unit factor cannot fit signed int64.
    if num > i64::MAX as i128 || den > i64::MAX as i128 {
        return Err(unsupported());
    }
    let max = i64::MAX as i128;
    if value > max / num || value < -((max - den + 1) / num) {
        return Err(overflow());
    }
    Ok((value * num).div_euclid(den) as i64)
}

fn days_from_month(month: i128) -> i128 {
    let year = 1970 + month.div_euclid(12);
    let month = month.rem_euclid(12) + 1;
    let year = year - i128::from(month <= 2);
    let era = year.div_euclid(400);
    let yoe = year - era * 400;
    let mp = month + if month > 2 { -3 } else { 9 };
    era * 146097 + yoe * 365 + yoe / 4 - yoe / 100 + (153 * mp + 2) / 5 - 719468
}
fn month_from_days(days: i128) -> i128 {
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let month = mp + if mp < 10 { 3 } else { -9 };
    let year = yoe + era * 400 + i128::from(month <= 2);
    (year - 1970) * 12 + month - 1
}
pub fn convert(tick: i64, from: TimeUnit, to: TimeUnit, datetime: bool) -> Result<i64> {
    if from == to {
        return Ok(tick);
    }
    let coarse = from.base.min(to.base);
    let fine = from.base.max(to.base);
    if coarse != fine && fine > TimeBase::Month {
        let coarse = if coarse <= TimeBase::Month {
            TimeBase::Day
        } else {
            coarse
        };
        let factor = TimeUnit {
            base: coarse,
            step: 1,
        }
        .scale()
            / TimeUnit {
                base: fine,
                step: 1,
            }
            .scale();
        // The pinned NumPy conversion-factor builder reserves its top byte.
        if factor >= (1i128 << 56) {
            return Err(overflow());
        }
    }
    ratio(0, from.scale(), to.scale())?;
    if tick == NAT {
        return Ok(tick);
    }
    let value = tick as i128;
    if !datetime || (!from.calendar() && !to.calendar()) {
        return ratio(value, from.scale(), to.scale());
    }
    let months = if from.calendar() {
        value * from.step as i128 * if from.base == TimeBase::Year { 12 } else { 1 }
    } else {
        let day = TimeUnit {
            base: TimeBase::Day,
            step: 1,
        };
        month_from_days(ratio(value, from.scale(), day.scale())? as i128)
    };
    if to.calendar() {
        return Ok(months
            .div_euclid(to.step as i128 * if to.base == TimeBase::Year { 12 } else { 1 })
            as i64);
    }
    ratio(
        days_from_month(months),
        TimeUnit {
            base: TimeBase::Day,
            step: 1,
        }
        .scale(),
        to.scale(),
    )
}

pub fn overflow() -> crate::error::Error {
    crate::error::Error::new(
        crate::error::ErrorCode::DTypeUnsupported,
        "Temporal arithmetic or unit conversion overflows int64. Use a coarser unit or smaller \
         values.",
    )
}
pub fn checked_tick(value: Option<i64>) -> Result<i64> {
    value.filter(|&x| x != NAT).ok_or_else(overflow)
}

pub fn milliseconds(tick: i64, from: TimeUnit) -> Result<i64> {
    let value = if from.calendar() {
        let months =
            tick as i128 * from.step as i128 * if from.base == TimeBase::Year { 12 } else { 1 };
        days_from_month(months) * 86400000
    } else {
        let ms = TimeUnit::parse("ms")?;
        let (mut a, mut b) = (from.scale(), ms.scale());
        while b != 0 {
            (a, b) = (b, a % b);
        }
        (tick as i128 * (from.scale() / a)).div_euclid(ms.scale() / a)
    };
    if !(-8640000000000000..=8640000000000000).contains(&value) {
        return Err(overflow());
    }
    Ok(value as i64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Buffer, DType, Scalar,
        array::NdArray,
        buffer::DenseArray,
        kernels::{self, BinaryOp},
        memory::{Device, MemoryBudget},
        shape::Selector,
        sync::CancellationToken,
    };
    fn unit(s: &str) -> TimeUnit {
        TimeUnit::parse(s).unwrap()
    }
    #[test]
    fn exact_units_calendar_negative_ticks_and_overflow() {
        assert_eq!(convert(-1, unit("ns"), unit("s"), true).unwrap(), -1);
        assert_eq!(convert(-1, unit("Y"), unit("D"), true).unwrap(), -365);
        assert_eq!(convert(-1, unit("Y"), unit("D"), false).unwrap(), -366);
        assert_eq!(convert(-13, unit("M"), unit("D"), true).unwrap(), -396);
        assert_eq!(convert(NAT, unit("ns"), unit("ms"), false).unwrap(), NAT);
        assert!(convert(i64::MIN + 1, unit("ns"), unit("ms"), false).is_err());
        assert!(convert(i64::MAX, unit("s"), unit("ns"), true).is_err());
        assert!(convert(0, unit("ps"), unit("M"), true).is_err());
        assert_eq!(unit("3s").common(unit("2ms")).unwrap(), unit("2ms"));
        for s in ["0s", "-1s", "2147483648s", "nsx", ""] {
            assert!(TimeUnit::parse(s).is_err());
        }
    }
    #[test]
    fn metadata_and_unicode_storage_follow_view_lifetime_and_budget() {
        let budget = MemoryBudget::new(Device::Cpu, 32);
        let source = Buffer::Unicode {
            width: 2,
            values: vec!['😀' as u32, 0, '한' as u32, '글' as u32],
        };
        let array = NdArray::from_dense(
            1,
            DenseArray::new(vec![2], source, budget.reserve(16).unwrap()).unwrap(),
        );
        let view = array
            .view(&[Selector::Slice {
                start: None,
                stop: None,
                step: Some(-1),
            }])
            .unwrap();
        assert_eq!(view.get_scalar(&[-1]).unwrap(), Scalar::Text("😀".into()));
        let copy = view.snapshot(&budget).unwrap();
        assert_eq!(copy.dtype(), DType::Str(2));
        assert_eq!(budget.usage().used, 32);
        assert!(view.snapshot(&budget).is_err());
        drop(copy);
        drop(array);
        assert_eq!(budget.usage().used, 16);
        drop(view);
        assert_eq!(budget.usage().used, 0);
    }
    #[test]
    fn cancelled_kernels_and_overflow_release_reserved_results() {
        let budget = MemoryBudget::new(Device::Cpu, 64);
        let dtype = DType::Timedelta64(unit("ns"));
        let a = DenseArray::new(
            vec![1],
            Buffer::Temporal {
                dtype,
                ticks: vec![i64::MAX],
            },
            budget.reserve(8).unwrap(),
        )
        .unwrap();
        let b = DenseArray::new(
            vec![1],
            Buffer::Temporal {
                dtype,
                ticks: vec![1],
            },
            budget.reserve(8).unwrap(),
        )
        .unwrap();
        let token = CancellationToken::new();
        assert!(kernels::binary(BinaryOp::Add, &a, &b, dtype, &budget, &token).is_err());
        assert_eq!(budget.usage().used, 16);
        token.cancel();
        assert!(kernels::binary(BinaryOp::Sub, &a, &b, dtype, &budget, &token).is_err());
        assert_eq!(budget.usage().used, 16);
        drop(a);
        drop(b);
        assert_eq!(budget.usage().used, 0);
    }
}
