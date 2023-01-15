use core::hash::Hash;
use core::{
    fmt,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, Flags, SerializationError, Valid, Validate,
};
use ff::Field;
use num_traits::{One, Zero};
use std::iter;
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
        Fp(self.0.add(rhs.0))
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

impl Valid for Fp {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl ark_serialize::CanonicalDeserialize for Fp {
    fn deserialize_with_mode<R: ark_serialize::Read>(
        mut reader: R,
        _compress: Compress,
        _validate: Validate,
    ) -> Result<Self, SerializationError> {
        let mut buff = [0u8; 48];
        reader
            .read(&mut buff[..])
            .map_err(|_| SerializationError::InvalidData)?;
        Option::from(blstrs::Fp::from_bytes_le(&buff).map(|f| Fp(f)))
            .ok_or(SerializationError::InvalidData)
    }
}

impl ark_serialize::CanonicalDeserializeWithFlags for Fp {
    fn deserialize_with_flags<R: ark_serialize::Read, F: Flags>(
        reader: R,
    ) -> Result<(Self, F), SerializationError> {
        Self::deserialize_with_mode(reader, Compress::No, Validate::No)
            .map(|s| (s, F::from_u8(0).unwrap()))
    }
}

impl ark_serialize::CanonicalSerialize for Fp {
    fn serialize_with_mode<W: ark_serialize::Write>(
        &self,
        mut writer: W,
        _compress: Compress,
    ) -> Result<(), SerializationError> {
        writer
            .write(&self.0.to_bytes_le()[..])
            .map(|_| ())
            .map_err(|_| SerializationError::InvalidData)
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        48
    }
}

impl ark_serialize::CanonicalSerializeWithFlags for Fp {
    fn serialize_with_flags<W: ark_serialize::Write, F: Flags>(
        &self,
        writer: W,
        _flags: F,
    ) -> Result<(), SerializationError> {
        Self::serialize_with_mode(&self, writer, Compress::No)
    }

    fn serialized_size_with_flags<F: Flags>(&self) -> usize {
        Self::serialized_size(&self, Compress::No)
    }
}

//impl ark_ff::Field for Fp {
//    type BasePrimeField = Fp;
//
//    type BasePrimeFieldIter = iter::Once<Self::BasePrimeField>;
//
//    const SQRT_PRECOMP: Option<ark_ff::SqrtPrecomputation<Self>> = None;
//
//    const ZERO: Self = Fp::zero();
//
//    const ONE: Self = Fp::one();
//
//    fn extension_degree() -> u64 {
//        1
//    }
//
//    fn to_base_prime_field_elements(&self) -> Self::BasePrimeFieldIter {
//        iter::once(*self)
//    }
//
//    fn from_base_prime_field_elems(elems: &[Self::BasePrimeField]) -> Option<Self> {
//        if elems.len() != (Self::extension_degree() as usize) {
//            return None;
//        }
//        Some(elems[0])
//    }
//
//    fn from_base_prime_field(elem: Self::BasePrimeField) -> Self {
//        elem
//    }
//
//    #[inline]
//    fn double(&self) -> Self {
//        Fp(self.0.double())
//    }
//
//    fn double_in_place(&mut self) -> &mut Self {
//        self.0 = self.0.double()
//    }
//
//    fn neg_in_place(&mut self) -> &mut Self {
//        self.0 = self.0.neg();
//    }
//
//    fn from_random_bytes_with_flags<F: Flags>(bytes: &[u8]) -> Option<(Self, F)> {
//        let blst_buffer = [0u8; 48];
//        blst_buffer[..].copy_from_slice(bytes);
//        blstrs::Fp::from_bytes_le(&blst_buffer);
//    }
//
//    fn legendre(&self) -> ark_ff::LegendreSymbol {
//        todo!()
//    }
//
//    fn square(&self) -> Self {
//        Fp(self.0.square())
//    }
//
//    fn square_in_place(&mut self) -> &mut Self {
//        self.0 = self.0.square();
//    }
//
//    fn inverse(&self) -> Option<Self> {
//        self.0.invert()
//    }
//
//    fn inverse_in_place(&mut self) -> Option<&mut Self> {
//        self.0 = self.0.invert();
//    }
//
//    fn frobenius_map_in_place(&mut self, power: usize) {
//        todo!()
//    }
//
//    fn characteristic() -> &'static [u64] {
//        Self::BasePrimeField::characteristic()
//    }
//
//    fn from_random_bytes(bytes: &[u8]) -> Option<Self> {
//        Self::from_random_bytes_with_flags::<EmptyFlags>(bytes).map(|f| f.0)
//    }
//
//    fn sqrt(&self) -> Option<Self> {
//        match Self::SQRT_PRECOMP {
//            Some(tv) => tv.sqrt(self),
//            None => self.0.sqrt(),
//        }
//    }
//
//    fn sqrt_in_place(&mut self) -> Option<&mut Self> {
//        (*self).sqrt().map(|sqrt| {
//            *self = sqrt;
//            self
//        })
//    }
//
//    fn sum_of_products<const T: usize>(a: &[Self; T], b: &[Self; T]) -> Self {
//        let mut sum = Self::zero();
//        for i in 0..a.len() {
//            sum += a[i] * b[i];
//        }
//        sum
//    }
//
//    fn frobenius_map(&self, power: usize) -> Self {
//        let mut this = *self;
//        this.frobenius_map_in_place(power);
//        this
//    }
//
//    fn pow<S: AsRef<[u64]>>(&self, exp: S) -> Self {
//        let mut res = Self::one();
//
//        for i in ark_ff::BitIteratorBE::without_leading_zeros(exp) {
//            res.square_in_place();
//
//            if i {
//                res *= self;
//            }
//        }
//        res
//    }
//
//    fn pow_with_table<S: AsRef<[u64]>>(powers_of_2: &[Self], exp: S) -> Option<Self> {
//        let mut res = Self::one();
//        for (pow, bit) in ark_ff::BitIteratorLE::without_trailing_zeros(exp).enumerate() {
//            if bit {
//                res *= powers_of_2.get(pow)?;
//            }
//        }
//        Some(res)
//    }
//}

#[cfg(test)]
mod tests {
    use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
    use ff::Field;

    use super::{Fp, One};

    #[test]
    fn fp_playground() {
        let fp1: Fp = Fp::one();
        let fp2: Fp = Fp::one();
        let f2: Fp = fp1 + fp2;
        let expf2: Fp = Fp(blstrs::Fp::from(2u64));
        assert_eq!(f2, expf2);

        //let f41: Fp = f2.double();
        let f42: Fp = f2 * f2;
        let expf4: Fp = Fp(blstrs::Fp::from(4u64));
        assert_eq!(f42, expf4);
        assert_eq!(f42, f42);

        let mut buff = Vec::new();
        expf4.serialize_compressed(&mut buff).unwrap();
        let readf4 = Fp::deserialize_compressed(&buff[..]).unwrap();
        assert_eq!(expf4, readf4);
    }
}
