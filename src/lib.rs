use core::hash::Hash;
use core::{
    fmt,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};
use ff::Field;
use num_traits::{One, Zero};
use std::{hash::Hasher, ops::Deref};
use zeroize::Zeroize;

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Fp(blstrs::Fp);

impl Deref for Fp {
    type Target = blstrs::Fp;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl fmt::Display for Fp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

impl AddAssign<&Fp> for Fp {
    #[inline]
    fn add_assign(&mut self, rhs: &Fp) {
        self.0.add_assign(rhs.0)
    }
}

impl AddAssign<Fp> for Fp {
    #[inline]
    fn add_assign(&mut self, rhs: Fp) {
        self.0.add_assign(rhs.0)
    }
}

impl Add<Fp> for Fp {
    type Output = Fp;

    #[inline]
    fn add(mut self, rhs: Fp) -> Fp {
        self += &rhs;
        self
    }
}

impl Add<&Fp> for &Fp {
    type Output = Fp;

    #[inline]
    fn add(self, rhs: &Fp) -> Fp {
        let mut out = *self;
        out += rhs;
        out
    }
}

impl Add<&Fp> for Fp {
    type Output = Fp;
    #[inline]
    fn add(self, rhs: &Fp) -> Fp {
        self.0.add(rhs.0);
        self
    }
}
impl Neg for &Fp {
    type Output = Fp;

    #[inline]
    fn neg(self) -> Fp {
        Fp(self.0.neg())
    }
}

impl Neg for Fp {
    type Output = Fp;

    #[inline]
    fn neg(self) -> Fp {
        Fp(self.0.neg())
    }
}

impl SubAssign<&Fp> for Fp {
    #[inline]
    fn sub_assign(&mut self, rhs: &Fp) {
        self.0.sub_assign(rhs.0);
    }
}
impl Sub<&Fp> for &Fp {
    type Output = Fp;

    #[inline]
    fn sub(self, rhs: &Fp) -> Fp {
        let mut out = *self;
        out.0 -= rhs.0;
        out
    }
}

impl MulAssign<&Fp> for Fp {
    #[inline]
    fn mul_assign(&mut self, rhs: &Fp) {
        self.0.mul_assign(rhs.0);
    }
}

impl Mul<&Fp> for Fp {
    type Output = Fp;

    #[inline]
    fn mul(mut self, rhs: &Fp) -> Fp {
        self.0.mul_assign(rhs.0);
        self
    }
}

impl Mul<Fp> for Fp {
    type Output = Fp;
    #[inline]
    fn mul(mut self, rhs: Fp) -> Fp {
        self.0.mul_assign(rhs.0);
        self
    }
}

impl Mul<&Fp> for Fp {
    type Output = Fp;

    #[inline]
    fn mul(mut self, rhs: &Fp) -> Fp {
        self.0.mul_assign(rhs.0);
        self
    }
}
impl Mul<&Fp> for &Fp {
    type Output = Fp;

    #[inline]
    fn mul(self, rhs: &Fp) -> Fp {
        let mut out = *self;
        out *= rhs;
        out
    }
}

impl One for Fp {
    fn one() -> Self {
        Fp(blstrs::Fp::one())
    }
}

impl Zero for Fp {
    fn zero() -> Self {
        Fp(blstrs::Fp::zero())
    }
    fn is_zero(&self) -> bool {
        self.0.is_zero().into()
    }
}

impl Zeroize for Fp {
    fn zeroize(&mut self) {
        self.0 = blstrs::Fp::from(0);
    }
}
// TODO check invariant
#[allow(clippy::derive_hash_xor_eq)]
impl Hash for Fp {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write(&self.0.to_bytes_le()[..]);
    }
}

impl ark_ff::Field for Fp {
    type BasePrimeField = Fp;

    type BasePrimeFieldIter;

    const SQRT_PRECOMP: Option<ark_ff::SqrtPrecomputation<Self>> = None;

    const ZERO: Self = Fp::zero();

    const ONE: Self = Fp::one();

    fn extension_degree() -> u64 {
        1
    }

    fn to_base_prime_field_elements(&self) -> Self::BasePrimeFieldIter {
        todo!()
    }

    fn from_base_prime_field_elems(elems: &[Self::BasePrimeField]) -> Option<Self> {
        todo!()
    }

    fn from_base_prime_field(elem: Self::BasePrimeField) -> Self {
        elem
    }

    fn double(&self) -> Self {
        Fp(self.0.double())
    }

    fn double_in_place(&mut self) -> &mut Self {
        self.0 = self.0.double()
    }

    fn neg_in_place(&mut self) -> &mut Self {
        self.0 = self.0.neg();
    }

    fn from_random_bytes_with_flags<F: Flags>(bytes: &[u8]) -> Option<(Self, F)> {
        todo!()
    }

    fn legendre(&self) -> ark_ff::LegendreSymbol {
        todo!()
    }

    fn square(&self) -> Self {
        Fp(self.0.square())
    }

    fn square_in_place(&mut self) -> &mut Self {
        self.0 = self.0.square();
    }

    fn inverse(&self) -> Option<Self> {
        self.0.invert()
    }

    fn inverse_in_place(&mut self) -> Option<&mut Self> {
        self.0 = self.0.invert();
    }

    fn frobenius_map_in_place(&mut self, power: usize) {
        todo!()
    }

    fn characteristic() -> &'static [u64] {
        Self::BasePrimeField::characteristic()
    }

    fn from_random_bytes(bytes: &[u8]) -> Option<Self> {
        Self::from_random_bytes_with_flags::<EmptyFlags>(bytes).map(|f| f.0)
    }

    fn sqrt(&self) -> Option<Self> {
        match Self::SQRT_PRECOMP {
            Some(tv) => tv.sqrt(self),
            None => self.0.sqrt(),
        }
    }

    fn sqrt_in_place(&mut self) -> Option<&mut Self> {
        (*self).sqrt().map(|sqrt| {
            *self = sqrt;
            self
        })
    }

    fn sum_of_products<const T: usize>(a: &[Self; T], b: &[Self; T]) -> Self {
        let mut sum = Self::zero();
        for i in 0..a.len() {
            sum += a[i] * b[i];
        }
        sum
    }

    fn frobenius_map(&self, power: usize) -> Self {
        let mut this = *self;
        this.frobenius_map_in_place(power);
        this
    }

    fn pow<S: AsRef<[u64]>>(&self, exp: S) -> Self {
        let mut res = Self::one();

        for i in ark_ff::BitIteratorBE::without_leading_zeros(exp) {
            res.square_in_place();

            if i {
                res *= self;
            }
        }
        res
    }

    fn pow_with_table<S: AsRef<[u64]>>(powers_of_2: &[Self], exp: S) -> Option<Self> {
        let mut res = Self::one();
        for (pow, bit) in ark_ff::BitIteratorLE::without_trailing_zeros(exp).enumerate() {
            if bit {
                res *= powers_of_2.get(pow)?;
            }
        }
        Some(res)
    }
}

#[cfg(test)]
mod tests {
    use super::{Fp, One};

    #[test]
    fn fp_playground() {
        let fp1 = Fp::one();
        let fp2 = Fp::one();
        let f2 = fp1 + fp2;
    }
}
