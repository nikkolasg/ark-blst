use crate::memory;
use core::hash::Hash;
use core::{
    fmt,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use ark_ff::BigInteger;
use std::ops::{Div, DivAssign};
use std::str::FromStr;

use ark_ff::{BigInt, PrimeField};
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
pub struct Scalar(pub(crate) blstrs::Scalar);

impl Deref for Scalar {
    type Target = blstrs::Scalar;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

impl AddAssign<&Scalar> for Scalar {
    #[inline]
    fn add_assign(&mut self, rhs: &Scalar) {
        self.0.add_assign(rhs.0)
    }
}

impl AddAssign<Scalar> for Scalar {
    #[inline]
    fn add_assign(&mut self, rhs: Scalar) {
        self.0.add_assign(rhs.0)
    }
}

impl<'a> AddAssign<&'a mut Scalar> for Scalar {
    fn add_assign(&mut self, rhs: &'a mut Scalar) {
        self.0.add_assign(rhs.0);
    }
}

impl Add<Scalar> for Scalar {
    type Output = Scalar;

    #[inline]
    fn add(self, rhs: Scalar) -> Scalar {
        Scalar(self.0 + rhs.0)
    }
}

impl Add<&Scalar> for &Scalar {
    type Output = Scalar;

    #[inline]
    fn add(self, rhs: &Scalar) -> Scalar {
        Scalar(self.0 + rhs.0)
    }
}

impl Add<&Scalar> for Scalar {
    type Output = Scalar;
    #[inline]
    fn add(self, rhs: &Scalar) -> Scalar {
        Scalar(self.0 + rhs.0)
    }
}
impl Neg for &Scalar {
    type Output = Scalar;

    #[inline]
    fn neg(self) -> Scalar {
        Scalar(self.0.neg())
    }
}

impl Neg for Scalar {
    type Output = Scalar;

    #[inline]
    fn neg(self) -> Scalar {
        Scalar(self.0.neg())
    }
}

impl SubAssign<&Scalar> for Scalar {
    #[inline]
    fn sub_assign(&mut self, rhs: &Scalar) {
        self.0.sub_assign(rhs.0);
    }
}

impl<'a> SubAssign<&'a mut Scalar> for Scalar {
    fn sub_assign(&mut self, rhs: &'a mut Scalar) {
        self.0.sub_assign(rhs.0);
    }
}

impl SubAssign<Scalar> for Scalar {
    fn sub_assign(&mut self, rhs: Scalar) {
        self.0.sub_assign(rhs.0)
    }
}

impl Sub<&Scalar> for &Scalar {
    type Output = Scalar;

    #[inline]
    fn sub(self, rhs: &Scalar) -> Scalar {
        Scalar(self.0 - rhs.0)
    }
}

impl MulAssign<&Scalar> for Scalar {
    #[inline]
    fn mul_assign(&mut self, rhs: &Scalar) {
        self.0.mul_assign(rhs.0);
    }
}

impl MulAssign<Scalar> for Scalar {
    fn mul_assign(&mut self, rhs: Self) {
        self.0.mul_assign(rhs.0);
    }
}

impl<'a> MulAssign<&'a mut Scalar> for Scalar {
    fn mul_assign(&mut self, rhs: &'a mut Scalar) {
        self.0.mul_assign(rhs.0);
    }
}

impl Mul<&Scalar> for Scalar {
    type Output = Scalar;

    #[inline]
    fn mul(self, rhs: &Scalar) -> Scalar {
        Scalar(self.0 * rhs.0)
    }
}

impl Mul<Scalar> for Scalar {
    type Output = Scalar;
    #[inline]
    fn mul(self, rhs: Scalar) -> Scalar {
        Scalar(self.0 * rhs.0)
    }
}

impl Mul<&Scalar> for &Scalar {
    type Output = Scalar;

    #[inline]
    fn mul(self, rhs: &Scalar) -> Scalar {
        Scalar(self.0 * rhs.0)
    }
}

impl<'a> Mul<&'a mut Scalar> for Scalar {
    type Output = Scalar;

    fn mul(self, rhs: &'a mut Scalar) -> Self::Output {
        Scalar(self.0 * rhs.0)
    }
}

impl One for Scalar {
    fn one() -> Self {
        Scalar(blstrs::Scalar::one())
    }
}

impl Zero for Scalar {
    fn zero() -> Self {
        Scalar(blstrs::Scalar::zero())
    }
    fn is_zero(&self) -> bool {
        self.0.is_zero().into()
    }
}

impl Zeroize for Scalar {
    fn zeroize(&mut self) {
        self.0 = blstrs::Scalar::from(0);
    }
}
// TODO check invariant
#[allow(unknown_lints, renamed_and_removed_lints)]
#[allow(clippy::derived_hash_with_manual_eq, clippy::derive_hash_xor_eq)]
impl Hash for Scalar {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write(&self.0.to_bytes_le()[..]);
    }
}

impl Valid for Scalar {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

#[allow(clippy::redundant_closure)]
impl ark_serialize::CanonicalDeserialize for Scalar {
    fn deserialize_with_mode<R: ark_serialize::Read>(
        mut reader: R,
        _compress: Compress,
        _validate: Validate,
    ) -> Result<Self, SerializationError> {
        let mut buff = [0u8; 32];
        reader
            .read(&mut buff[..])
            .map_err(|_| SerializationError::InvalidData)?;
        Option::from(blstrs::Scalar::from_bytes_le(&buff).map(|f| Scalar(f)))
            .ok_or(SerializationError::InvalidData)
    }
}

impl ark_serialize::CanonicalDeserializeWithFlags for Scalar {
    fn deserialize_with_flags<R: ark_serialize::Read, F: Flags>(
        reader: R,
    ) -> Result<(Self, F), SerializationError> {
        Self::deserialize_with_mode(reader, Compress::No, Validate::No)
            .map(|s| (s, F::from_u8(0).unwrap()))
    }
}

impl ark_serialize::CanonicalSerialize for Scalar {
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
        32
    }
}

impl ark_serialize::CanonicalSerializeWithFlags for Scalar {
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

impl From<bool> for Scalar {
    fn from(value: bool) -> Self {
        match value {
            true => Scalar::from(1u64),
            false => Scalar::from(0u64),
        }
    }
}

macro_rules! impl_from {
    ($t:ty) => {
        impl From<$t> for Scalar {
            fn from(value: $t) -> Self {
                Scalar(blstrs::Scalar::from(value as u64))
            }
        }
    };
}

impl_from!(u8);
impl_from!(u16);
impl_from!(u32);
impl_from!(u64);
impl_from!(u128);

impl<'a> core::iter::Product<&'a Scalar> for Scalar {
    fn product<I: Iterator<Item = &'a Scalar>>(iter: I) -> Self {
        iter.fold(Scalar::one(), |acc, x| acc * x)
    }
}

impl core::iter::Product<Scalar> for Scalar {
    fn product<I: Iterator<Item = Scalar>>(iter: I) -> Self {
        iter.fold(Scalar::one(), |acc, x| acc * x)
    }
}

impl<'a> core::iter::Sum<&'a Scalar> for Scalar {
    fn sum<I: Iterator<Item = &'a Scalar>>(iter: I) -> Self {
        iter.fold(Scalar::zero(), |acc, x| acc + x)
    }
}

impl core::iter::Sum<Scalar> for Scalar {
    fn sum<I: Iterator<Item = Scalar>>(iter: I) -> Self {
        iter.fold(Scalar::zero(), |acc, x| acc + x)
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Div<Scalar> for Scalar {
    type Output = Scalar;

    fn div(self, rhs: Scalar) -> Self::Output {
        self * Scalar(rhs.0.invert().unwrap())
    }
}

impl<'a> Div<&'a mut Scalar> for Scalar {
    type Output = Scalar;

    fn div(self, rhs: &'a mut Scalar) -> Self::Output {
        let mut c = self;
        c.div_assign(rhs);
        c
    }
}
impl<'a> Div<&'a Scalar> for Scalar {
    type Output = Scalar;

    fn div(self, rhs: &'a Scalar) -> Self::Output {
        let mut c = self;
        c.div_assign(rhs);
        c
    }
}

#[allow(clippy::suspicious_op_assign_impl)]
impl DivAssign for Scalar {
    fn div_assign(&mut self, rhs: Self) {
        *self *= Scalar(rhs.0.invert().unwrap());
    }
}

#[allow(clippy::suspicious_op_assign_impl)]
impl<'a> DivAssign<&'a Scalar> for Scalar {
    fn div_assign(&mut self, rhs: &'a Scalar) {
        *self *= Scalar(rhs.0.invert().unwrap());
    }
}

#[allow(clippy::suspicious_op_assign_impl)]
impl<'a> DivAssign<&'a mut Scalar> for Scalar {
    fn div_assign(&mut self, rhs: &'a mut Scalar) {
        *self *= Scalar(rhs.0.invert().unwrap());
    }
}

impl<'a> Add<&'a mut Scalar> for Scalar {
    type Output = Scalar;

    fn add(self, rhs: &'a mut Scalar) -> Self::Output {
        Scalar(self.0 + rhs.0)
    }
}

impl Sub<Scalar> for Scalar {
    type Output = Scalar;

    fn sub(self, rhs: Scalar) -> Self::Output {
        Scalar(self.0 - rhs.0)
    }
}

impl<'a> Sub<&'a Scalar> for Scalar {
    type Output = Scalar;

    fn sub(self, rhs: &'a Scalar) -> Self::Output {
        Scalar(self.0 - rhs.0)
    }
}

impl<'a> Sub<&'a mut Scalar> for Scalar {
    type Output = Scalar;

    fn sub(self, rhs: &'a mut Scalar) -> Self::Output {
        Scalar(self.0 - rhs.0)
    }
}

impl ark_ff::UniformRand for Scalar {
    fn rand<R: ark_std::rand::Rng + ?Sized>(rng: &mut R) -> Self {
        Scalar(blstrs::Scalar::random(rng))
    }
}
impl From<num_bigint::BigUint> for Scalar {
    fn from(value: num_bigint::BigUint) -> Self {
        Scalar(
            blstrs::Scalar::from_bytes_le(memory::slice_to_constant_size(&value.to_bytes_le()))
                .unwrap(),
        )
    }
}

impl From<BigInt<4>> for Scalar {
    fn from(value: BigInt<4>) -> Self {
        Scalar(
            blstrs::Scalar::from_bytes_le(memory::slice_to_constant_size(&value.to_bytes_le()))
                .unwrap(),
        )
    }
}

impl FromStr for Scalar {
    type Err = SerializationError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let ark_fq = ark_bls12_381::Fr::from_str(s).map_err(|_| SerializationError::InvalidData)?;
        Ok(Scalar::from(ark_fq.into_bigint()))
    }
}

impl From<ark_bls12_381::Fr> for Scalar {
    fn from(value: ark_bls12_381::Fr) -> Self {
        Scalar::from(value.into_bigint())
    }
}
impl From<Scalar> for num_bigint::BigUint {
    fn from(value: Scalar) -> Self {
        let slice = value.0.to_bytes_le();
        let s = memory::constant_size_to_slice(&slice);
        num_bigint::BigUint::from_bytes_le(s)
    }
}

impl From<Scalar> for <Scalar as PrimeField>::BigInt {
    fn from(value: Scalar) -> Self {
        let bg: num_bigint::BigUint = value.into();
        BigInt::<4>::try_from(bg).unwrap()
    }
}

impl ark_ff::FftField for Scalar {
    const GENERATOR: Self = Scalar(blstrs::scalar::R);

    const TWO_ADICITY: u32 = blstrs::scalar::S;

    const TWO_ADIC_ROOT_OF_UNITY: Self = Scalar(blstrs::scalar::ROOT_OF_UNITY);
}

impl ark_ff::PrimeField for Scalar {
    type BigInt = ark_ff::BigInteger256;

    const MODULUS: Self::BigInt = ark_ff::BigInt::<4>([
        0xffff_ffff_0000_0001,
        0x53bd_a402_fffe_5bfe,
        0x3339_d808_09a1_d805,
        0x73ed_a753_299d_7d48,
    ]);

    // TODO : currently using bls12-381 for quick and dirty - best to copy the constants here
    const MODULUS_MINUS_ONE_DIV_TWO: Self::BigInt =
        <ark_bls12_381::Fr as ark_ff::PrimeField>::MODULUS_MINUS_ONE_DIV_TWO;

    const MODULUS_BIT_SIZE: u32 = 255;

    const TRACE: Self::BigInt = <ark_bls12_381::Fr as ark_ff::PrimeField>::TRACE;

    const TRACE_MINUS_ONE_DIV_TWO: Self::BigInt =
        <ark_bls12_381::Fr as ark_ff::PrimeField>::TRACE_MINUS_ONE_DIV_TWO;

    fn from_bigint(repr: Self::BigInt) -> Option<Self> {
        Some(Scalar(
            blstrs::Scalar::from_bytes_le(memory::slice_to_constant_size(
                repr.to_bytes_le().as_slice(),
            ))
            .unwrap(),
        ))
    }

    fn into_bigint(self) -> Self::BigInt {
        self.into()
    }
}

impl ark_ff::Field for Scalar {
    type BasePrimeField = Scalar;

    type BasePrimeFieldIter = iter::Once<Self::BasePrimeField>;

    const SQRT_PRECOMP: Option<ark_ff::SqrtPrecomputation<Self>> = None;

    const ZERO: Self = Scalar(blstrs::scalar::ZERO);

    const ONE: Self = Scalar(blstrs::scalar::R);

    fn extension_degree() -> u64 {
        1
    }

    fn to_base_prime_field_elements(&self) -> Self::BasePrimeFieldIter {
        iter::once(*self)
    }

    fn from_base_prime_field_elems(elems: &[Self::BasePrimeField]) -> Option<Self> {
        if elems.len() != (Self::extension_degree() as usize) {
            return None;
        }
        Some(elems[0])
    }

    fn from_base_prime_field(elem: Self::BasePrimeField) -> Self {
        elem
    }

    #[inline]
    fn double(&self) -> Self {
        Scalar(self.0.double())
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
        let blst_buffer: &[u8; 32] = memory::slice_to_constant_size(bytes);
        blstrs::Scalar::from_bytes_le(blst_buffer)
            .map(|fp| (Scalar(fp), F::from_u8(0).unwrap()))
            .into()
    }

    fn legendre(&self) -> ark_ff::LegendreSymbol {
        todo!()
    }

    fn square(&self) -> Self {
        Scalar(self.0.square())
    }

    fn square_in_place(&mut self) -> &mut Self {
        unimplemented!()
    }

    #[allow(clippy::redundant_closure)]
    fn inverse(&self) -> Option<Self> {
        self.0.invert().map(|f| Scalar(f)).into()
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
        blstrs::scalar::MODULUS[..].as_ref()
    }

    fn from_random_bytes(bytes: &[u8]) -> Option<Self> {
        Self::from_random_bytes_with_flags::<EmptyFlags>(bytes).map(|f| f.0)
    }

    #[allow(clippy::redundant_closure)]
    fn sqrt(&self) -> Option<Self> {
        match Self::SQRT_PRECOMP {
            Some(tv) => tv.sqrt(self),
            None => self.0.sqrt().map(|f| Scalar(f)).into(),
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
        Self(self.0.pow_vartime(exp))
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

impl ark_crypto_primitives::sponge::Absorb for Scalar {
    fn to_sponge_bytes(&self, dest: &mut Vec<u8>) {
        let buff = self.0.to_bytes_le();
        dest.copy_from_slice(&buff[..]);
    }

    fn to_sponge_field_elements<F: PrimeField>(&self, dest: &mut Vec<F>) {
        // TODO do this manually
        ark_bls12_381::Fr::from_bigint(self.into_bigint()).to_sponge_field_elements(dest);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp_tests() {
        crate::tests::field_test::<Scalar>();
    }
}
