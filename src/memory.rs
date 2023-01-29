pub(crate) fn slice_to_constant_size<'a, const N: usize>(v: &'a [u8]) -> &'a [u8; N] {
    let ptr = v.as_ptr() as *const [u8; N];
    unsafe { &*ptr }
}

pub(crate) fn constant_size_to_slice<'a, const N: usize>(v: &'a [u8; N]) -> &'a [u8] {
    let ptr = v as *const [u8; N] as *const [u8];
    unsafe { &*ptr }
}
