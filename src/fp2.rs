use crate::fp::Fp;
use crate::memory;
use core::hash::Hash;
use core::{
    fmt,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use ark_ff::BigInteger;
use std::ops::{Div, DivAssign};

use ark_ff::BigInt;
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, EmptyFlags, Flags, SerializationError,
    Valid, Validate,
};
use ff::Field;
use num_traits::{One, Zero};
use std::iter;
use std::{hash::Hasher, ops::Deref};
use zeroize::Zeroize;

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Fp2(pub(crate) blstrs::Fp2);

impl Deref for Fp2 {
    type Target = blstrs::Fp2;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl fmt::Display for Fp2 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

impl AddAssign<&Fp2> for Fp2 {
    #[inline]
    fn add_assign(&mut self, rhs: &Fp2) {
        self.0.add_assign(rhs.0)
    }
}

impl AddAssign<Fp2> for Fp2 {
    #[inline]
    fn add_assign(&mut self, rhs: Fp2) {
        self.0.add_assign(rhs.0)
    }
}

impl<'a> AddAssign<&'a mut Fp2> for Fp2 {
    fn add_assign(&mut self, rhs: &'a mut Fp2) {
        self.0.add_assign(rhs.0);
    }
}

impl Add<Fp2> for Fp2 {
    type Output = Fp2;

    #[inline]
    fn add(self, rhs: Fp2) -> Fp2 {
        Fp2(self.0 + rhs.0)
    }
}

impl Add<&Fp2> for &Fp2 {
    type Output = Fp2;

    #[inline]
    fn add(self, rhs: &Fp2) -> Fp2 {
        Fp2(self.0 + rhs.0)
    }
}

impl Add<&Fp2> for Fp2 {
    type Output = Fp2;
    #[inline]
    fn add(self, rhs: &Fp2) -> Fp2 {
        Fp2(self.0 + rhs.0)
    }
}
impl Neg for &Fp2 {
    type Output = Fp2;

    #[inline]
    fn neg(self) -> Fp2 {
        Fp2(self.0.neg())
    }
}

impl Neg for Fp2 {
    type Output = Fp2;

    #[inline]
    fn neg(self) -> Fp2 {
        Fp2(self.0.neg())
    }
}

impl SubAssign<&Fp2> for Fp2 {
    #[inline]
    fn sub_assign(&mut self, rhs: &Fp2) {
        self.0.sub_assign(rhs.0);
    }
}

impl<'a> SubAssign<&'a mut Fp2> for Fp2 {
    fn sub_assign(&mut self, rhs: &'a mut Fp2) {
        self.0.sub_assign(rhs.0);
    }
}

impl SubAssign<Fp2> for Fp2 {
    fn sub_assign(&mut self, rhs: Fp2) {
        self.0.sub_assign(rhs.0)
    }
}

impl Sub<&Fp2> for &Fp2 {
    type Output = Fp2;

    #[inline]
    fn sub(self, rhs: &Fp2) -> Fp2 {
        Fp2(self.0 - rhs.0)
    }
}

impl MulAssign<&Fp2> for Fp2 {
    #[inline]
    fn mul_assign(&mut self, rhs: &Fp2) {
        self.0.mul_assign(rhs.0);
    }
}

impl MulAssign<Fp2> for Fp2 {
    fn mul_assign(&mut self, rhs: Self) {
        self.0.mul_assign(rhs.0);
    }
}

impl<'a> MulAssign<&'a mut Fp2> for Fp2 {
    fn mul_assign(&mut self, rhs: &'a mut Fp2) {
        self.0.mul_assign(rhs.0);
    }
}

impl Mul<&Fp2> for Fp2 {
    type Output = Fp2;

    #[inline]
    fn mul(self, rhs: &Fp2) -> Fp2 {
        Fp2(self.0 * rhs.0)
    }
}

impl Mul<Fp2> for Fp2 {
    type Output = Fp2;
    #[inline]
    fn mul(self, rhs: Fp2) -> Fp2 {
        Fp2(self.0 * rhs.0)
    }
}

impl Mul<&Fp2> for &Fp2 {
    type Output = Fp2;

    #[inline]
    fn mul(self, rhs: &Fp2) -> Fp2 {
        Fp2(self.0 * rhs.0)
    }
}

impl<'a> Mul<&'a mut Fp2> for Fp2 {
    type Output = Fp2;

    fn mul(self, rhs: &'a mut Fp2) -> Self::Output {
        Fp2(self.0 * rhs.0)
    }
}

impl One for Fp2 {
    fn one() -> Self {
        Fp2(blstrs::Fp2::one())
    }
}

impl Zero for Fp2 {
    fn zero() -> Self {
        Fp2(blstrs::Fp2::zero())
    }
    fn is_zero(&self) -> bool {
        self.0.is_zero().into()
    }
}

impl Zeroize for Fp2 {
    fn zeroize(&mut self) {
        self.0 = blstrs::Fp2::from(0);
    }
}

// TODO check invariant
#[allow(clippy::derived_hash_with_manual_eq)]
impl Hash for Fp2 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write(&self.0.to_bytes_le()[..]);
    }
}

impl Valid for Fp2 {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

#[allow(clippy::redundant_closure)]
impl ark_serialize::CanonicalDeserialize for Fp2 {
    fn deserialize_with_mode<R: ark_serialize::Read>(
        mut reader: R,
        _compress: Compress,
        _validate: Validate,
    ) -> Result<Self, SerializationError> {
        let mut buff = [0u8; 96];
        reader
            .read(&mut buff[..])
            .map_err(|_| SerializationError::InvalidData)?;
        Option::from(blstrs::Fp2::from_bytes_le(&buff).map(|f| Fp2(f)))
            .ok_or(SerializationError::InvalidData)
    }
}

impl ark_serialize::CanonicalDeserializeWithFlags for Fp2 {
    fn deserialize_with_flags<R: ark_serialize::Read, F: Flags>(
        reader: R,
    ) -> Result<(Self, F), SerializationError> {
        Self::deserialize_with_mode(reader, Compress::No, Validate::No)
            .map(|s| (s, F::from_u8(0).unwrap()))
    }
}

impl ark_serialize::CanonicalSerialize for Fp2 {
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
        96
    }
}

impl ark_serialize::CanonicalSerializeWithFlags for Fp2 {
    fn serialize_with_flags<W: ark_serialize::Write, F: Flags>(
        &self,
        writer: W,
        _flags: F,
    ) -> Result<(), SerializationError> {
        Self::serialize_with_mode(self, writer, Compress::No)
    }

    fn serialized_size_with_flags<F: Flags>(&self) -> usize {
        Self::serialized_size(self, Compress::No)
    }
}

impl From<bool> for Fp2 {
    fn from(value: bool) -> Self {
        match value {
            true => Fp2::from(1u64),
            false => Fp2::from(0u64),
        }
    }
}

macro_rules! impl_from {
    ($t:ty) => {
        impl From<$t> for Fp2 {
            fn from(value: $t) -> Self {
                Fp2(blstrs::Fp2::from(value as u64))
            }
        }
    };
}

impl_from!(u8);
impl_from!(u16);
impl_from!(u32);
impl_from!(u64);

impl From<u128> for Fp2 {
    fn from(value: u128) -> Self {
        Fp2::new(Fp::from(value), Fp::zero())
    }
}

impl<'a> core::iter::Product<&'a Fp2> for Fp2 {
    fn product<I: Iterator<Item = &'a Fp2>>(iter: I) -> Self {
        iter.fold(Fp2::one(), |acc, x| acc * x)
    }
}

impl core::iter::Product<Fp2> for Fp2 {
    fn product<I: Iterator<Item = Fp2>>(iter: I) -> Self {
        iter.fold(Fp2::one(), |acc, x| acc * x)
    }
}

impl<'a> core::iter::Sum<&'a Fp2> for Fp2 {
    fn sum<I: Iterator<Item = &'a Fp2>>(iter: I) -> Self {
        iter.fold(Fp2::zero(), |acc, x| acc + x)
    }
}

impl core::iter::Sum<Fp2> for Fp2 {
    fn sum<I: Iterator<Item = Fp2>>(iter: I) -> Self {
        iter.fold(Fp2::zero(), |acc, x| acc + x)
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Div<Fp2> for Fp2 {
    type Output = Fp2;

    fn div(self, rhs: Fp2) -> Self::Output {
        self * Fp2(rhs.0.invert().unwrap())
    }
}

impl<'a> Div<&'a mut Fp2> for Fp2 {
    type Output = Fp2;

    fn div(self, rhs: &'a mut Fp2) -> Self::Output {
        let mut c = self;
        c.div_assign(rhs);
        c
    }
}
impl<'a> Div<&'a Fp2> for Fp2 {
    type Output = Fp2;

    fn div(self, rhs: &'a Fp2) -> Self::Output {
        let mut c = self;
        c.div_assign(rhs);
        c
    }
}

#[allow(clippy::suspicious_op_assign_impl)]
impl DivAssign for Fp2 {
    fn div_assign(&mut self, rhs: Self) {
        *self *= Fp2(rhs.0.invert().unwrap());
    }
}

#[allow(clippy::suspicious_op_assign_impl)]
impl<'a> DivAssign<&'a Fp2> for Fp2 {
    fn div_assign(&mut self, rhs: &'a Fp2) {
        *self *= Fp2(rhs.0.invert().unwrap());
    }
}

#[allow(clippy::suspicious_op_assign_impl)]
impl<'a> DivAssign<&'a mut Fp2> for Fp2 {
    fn div_assign(&mut self, rhs: &'a mut Fp2) {
        *self *= Fp2(rhs.0.invert().unwrap());
    }
}

impl<'a> Add<&'a mut Fp2> for Fp2 {
    type Output = Fp2;

    fn add(self, rhs: &'a mut Fp2) -> Self::Output {
        Fp2(self.0.add(rhs.0))
    }
}

impl Sub<Fp2> for Fp2 {
    type Output = Fp2;

    fn sub(self, rhs: Fp2) -> Self::Output {
        Fp2(self.0.sub(rhs.0))
    }
}

impl<'a> Sub<&'a Fp2> for Fp2 {
    type Output = Fp2;

    fn sub(self, rhs: &'a Fp2) -> Self::Output {
        Fp2(self.0.sub(rhs.0))
    }
}

impl<'a> Sub<&'a mut Fp2> for Fp2 {
    type Output = Fp2;

    fn sub(self, rhs: &'a mut Fp2) -> Self::Output {
        Fp2(self.0.sub(rhs.0))
    }
}

impl ark_ff::UniformRand for Fp2 {
    fn rand<R: ark_std::rand::Rng + ?Sized>(rng: &mut R) -> Self {
        Fp2(blstrs::Fp2::random(rng))
    }
}
impl From<num_bigint::BigUint> for Fp2 {
    fn from(value: num_bigint::BigUint) -> Self {
        Fp2(
            blstrs::Fp2::from_bytes_le(memory::slice_to_constant_size(&value.to_bytes_le()))
                .unwrap(),
        )
    }
}

impl From<BigInt<12>> for Fp2 {
    fn from(value: BigInt<12>) -> Self {
        Fp2(
            blstrs::Fp2::from_bytes_le(memory::slice_to_constant_size(&value.to_bytes_le()))
                .unwrap(),
        )
    }
}

impl From<ark_bls12_381::Fq2> for Fp2 {
    fn from(value: ark_bls12_381::Fq2) -> Self {
        let mut buff = Vec::new();
        value.serialize_compressed(&mut buff).unwrap();
        Fp2(blstrs::Fp2::from_bytes_le(memory::slice_to_constant_size(&buff)).unwrap())
    }
}
impl From<Fp2> for num_bigint::BigUint {
    fn from(value: Fp2) -> Self {
        let slice = value.0.to_bytes_le();
        let s = memory::constant_size_to_slice(&slice);
        num_bigint::BigUint::from_bytes_le(s)
    }
}

impl Fp2 {
    pub fn new(c0: Fp, c1: Fp) -> Self {
        Fp2(blstrs::Fp2::new(c0.0, c1.0))
    }
}
type BaseFieldIter<P> = <<P as ark_ff::Field>::BasePrimeField as ark_ff::Field>::BasePrimeFieldIter;
impl ark_ff::Field for Fp2 {
    type BasePrimeField = crate::fp::Fp;

    type BasePrimeFieldIter = iter::Chain<BaseFieldIter<Self>, BaseFieldIter<Self>>;

    const SQRT_PRECOMP: Option<ark_ff::SqrtPrecomputation<Self>> = None;

    const ZERO: Self = Fp2(blstrs::Fp2::new(blstrs::fp::ZERO, blstrs::fp::ZERO));

    const ONE: Self = Fp2(blstrs::Fp2::new(blstrs::fp::R, blstrs::fp::ZERO));

    fn extension_degree() -> u64 {
        Self::BasePrimeField::extension_degree() * 2
    }

    fn to_base_prime_field_elements(&self) -> Self::BasePrimeFieldIter {
        Fp(self.0.c0())
            .to_base_prime_field_elements()
            .chain(Fp(self.0.c1()).to_base_prime_field_elements())
    }

    fn from_base_prime_field_elems(elems: &[Self::BasePrimeField]) -> Option<Self> {
        if elems.len() != (Self::extension_degree() as usize) {
            return None;
        }
        let base_ext_deg = Self::BasePrimeField::extension_degree() as usize;
        Some(Self::new(
            Self::BasePrimeField::from_base_prime_field_elems(&elems[0..base_ext_deg]).unwrap(),
            Self::BasePrimeField::from_base_prime_field_elems(&elems[base_ext_deg..]).unwrap(),
        ))
    }

    fn from_base_prime_field(elem: Self::BasePrimeField) -> Self {
        Self::new(
            Self::BasePrimeField::from_base_prime_field(elem),
            Self::BasePrimeField::ZERO,
        )
    }

    #[inline]
    fn double(&self) -> Self {
        Fp2(self.0.double())
    }

    fn double_in_place(&mut self) -> &mut Self {
        self.0 = self.0.double();
        self
    }

    fn neg_in_place(&mut self) -> &mut Self {
        self.0 = self.0.neg();
        self
    }

    fn from_random_bytes_with_flags<F: Flags>(bytes: &[u8]) -> Option<(Self, F)> {
        let mut blst_buffer = [0u8; 96];
        blst_buffer[..].copy_from_slice(bytes);
        blstrs::Fp2::from_bytes_le(&blst_buffer)
            .map(|fp| (Fp2(fp), F::from_u8(0).unwrap()))
            .into()
    }

    fn legendre(&self) -> ark_ff::LegendreSymbol {
        todo!()
    }

    fn square(&self) -> Self {
        Fp2(self.0.square())
    }

    fn square_in_place(&mut self) -> &mut Self {
        self.0 = self.0.square();
        self
    }

    #[allow(clippy::redundant_closure)]
    fn inverse(&self) -> Option<Self> {
        self.0.invert().map(|f| Fp2(f)).into()
    }

    fn inverse_in_place(&mut self) -> Option<&mut Self> {
        self.0
            .invert()
            .map(|f| {
                self.0 = f;
                self
            })
            .into()
    }

    // nothing on base prime field
    fn frobenius_map_in_place(&mut self, _power: usize) {}

    fn characteristic() -> &'static [u64] {
        blstrs::fp::MODULUS[..].as_ref()
    }

    fn from_random_bytes(bytes: &[u8]) -> Option<Self> {
        Self::from_random_bytes_with_flags::<EmptyFlags>(bytes).map(|f| f.0)
    }

    #[allow(clippy::redundant_closure)]
    fn sqrt(&self) -> Option<Self> {
        match Self::SQRT_PRECOMP {
            Some(tv) => tv.sqrt(self),
            None => self.0.sqrt().map(|f| Fp2(f)).into(),
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
    use super::*;

    #[test]
    fn fp2_tests() {
        crate::tests::field_test::<Fp2>();
    }
}
