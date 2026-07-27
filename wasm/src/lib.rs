#[no_mangle]
pub extern "C" fn allocate_float_buffer(length: usize) -> *mut f32 {
    let mut values = Vec::<f32>::with_capacity(length);
    let pointer = values.as_mut_ptr();
    std::mem::forget(values);
    pointer
}

#[no_mangle]
pub unsafe extern "C" fn release_float_buffer(pointer: *mut f32, length: usize) {
    drop(Vec::from_raw_parts(pointer, 0, length));
}

#[no_mangle]
pub unsafe extern "C" fn apply_relu(pointer: *mut f32, length: usize) {
    let values = std::slice::from_raw_parts_mut(pointer, length);

    for value in values {
        *value = value.max(0.0);
    }
}

#[no_mangle]
pub unsafe extern "C" fn apply_linear(
    input_pointer: *const f32,
    input_count: usize,
    output_pointer: *mut f32,
    output_count: usize,
    weight_pointer: *const f32,
    bias_pointer: *const f32,
) {
    let inputs = std::slice::from_raw_parts(input_pointer, input_count);
    let outputs = std::slice::from_raw_parts_mut(output_pointer, output_count);
    let weights = std::slice::from_raw_parts(weight_pointer, input_count * output_count);
    let biases = std::slice::from_raw_parts(bias_pointer, output_count);

    for output_index in 0..output_count {
        let mut sum = biases[output_index];
        let weight_offset = output_index * input_count;

        for input_index in 0..input_count {
            sum += inputs[input_index] * weights[weight_offset + input_index];
        }

        outputs[output_index] = sum;
    }
}

