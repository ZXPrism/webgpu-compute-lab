import { BufferTypeEnum } from "./buffer_info";
import type { Kernel } from "./kernel";
import { KernelBuilder } from "./kernel_builder";

import reduce_shader from "./shaders/reduce_basic.wgsl?raw";
import reduce_native_shader from "./shaders/reduce_native.wgsl?raw";
import reduce_upsweep_shader from "./shaders/reduce_upsweep.wgsl?raw";
import reduce_flatten_shader from "./shaders/reduce_flatten.wgsl?raw";

import prefix_sum_native_shader from "./shaders/prefix_sum_native.wgsl?raw";
import prefix_sum_basic_shader from "./shaders/prefix_sum_basic.wgsl?raw";
import prefix_sum_cum_shader from "./shaders/prefix_sum_cum.wgsl?raw";
import prefix_sum_blelloch_shader from "./shaders/prefix_sum_blelloch.wgsl?raw";

let c_array_length = 1000000;
let c_reduce_kernel_segment_length = 256;
let c_prefix_sum_kernel_segment_length = 256;
let c_reduce_mode = 3; // 0: native; 1: basic; 2: upsweep; 3: flatten
let c_prefix_sum_mode = 1; // 0: native

let g_device: GPUDevice;

let g_prefix_sum_native_kernel: Kernel;

let g_prefix_sum_basic_kernel: Kernel;
let g_prefix_sum_cum_kernel: Kernel;

let g_reduce_native_kernel: Kernel;
let g_reduce_basic_kernel_chain: Kernel[] = [];
let g_reduce_basic_kernel_dispatch_params: number[] = [];

let g_reduce_upsweep_kernel_chain: Kernel[] = [];
let g_reduce_upsweep_kernel_dispatch_params: number[] = [];

let g_reduce_flatten_kernel_chain: Kernel[] = [];
let g_reduce_flatten_kernel_dispatch_params: number[] = [];

let g_array_length_buffer: GPUBuffer;
let g_input_array_buffer: GPUBuffer;
let g_input_array_sum: number = 0;

async function init_webgpu() {
    const adapter = await navigator.gpu?.requestAdapter();
    const device = await adapter?.requestDevice(
        {
            requiredFeatures: ["timestamp-query"]
        }
    );
    if (!device) {
        console.error("unable to initialize WebGPU");
        return;
    }

    const canvas = document.querySelector("canvas")!;
    const context = canvas.getContext("webgpu")!;
    context.configure({
        device,
        format: navigator.gpu.getPreferredCanvasFormat()
    })

    g_device = device;

    console.info("successfully initialized WebGPU");
}

function init_input_array() {
    g_array_length_buffer = g_device.createBuffer({
        label: 'array length buffer',
        size: 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const array_length_data = new Uint32Array(1);
    array_length_data[0] = c_array_length;
    g_device.queue.writeBuffer(g_array_length_buffer, 0, array_length_data.buffer);

    g_input_array_buffer = g_device.createBuffer({
        label: 'input array buffer',
        size: c_array_length * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const input_array_buffer = new Uint32Array(c_array_length);
    for (let i = 0; i < c_array_length; i++) {
        input_array_buffer[i] = Math.floor(Math.random() * 10);
    }
    g_device.queue.writeBuffer(g_input_array_buffer, 0, input_array_buffer.buffer);
    console.info("[input array]: ", input_array_buffer);

    {
        let t1 = performance.now();
        for (let i = 0; i < c_array_length; i++) {
            g_input_array_sum += input_array_buffer[i];
        }
        let t2 = performance.now();
        console.log("[reduce] cpu time: ", t2 - t1, "ms")
    }

    {
        const prefix_sum_array = new Uint32Array(c_array_length);
        let t1 = performance.now();
        prefix_sum_array[0] = input_array_buffer[0];
        for (let i = 1; i < c_array_length; i++) {
            prefix_sum_array[i] = prefix_sum_array[i - 1] + input_array_buffer[i];
        }
        let t2 = performance.now();

        console.info("[prefix sum (inclusive)]: ", prefix_sum_array);
        console.log("[prefix sum (inclusive)] cpu time: ", t2 - t1, "ms")
    }
}

function init_kernels_reduce() {
    // to reduce an input array with arbitrary size, we need multiple passes
    // current stretegy is to divide input array into segments, every segment is of equal size (256 elements), each workgroup process one segment
    // this is not optimal, should balance the segment size in each pass, as well as adding batching for each workgroup
    // but for simplicity I will stick to the former stretegy now

    if (c_reduce_mode == 0) {// reduce native
        const reduce_native_kernel_builder = new KernelBuilder(g_device, "reduce_native", reduce_native_shader, "compute");
        g_reduce_native_kernel = reduce_native_kernel_builder
            .add_buffer("array_length", 0, BufferTypeEnum.UNIFORM, g_array_length_buffer)
            .add_buffer("input_array", 1, BufferTypeEnum.READONLY_STORAGE, g_input_array_buffer)
            .create_then_add_buffer("output_sum", 2, BufferTypeEnum.STORAGE, 4)
            .build();
    } else if (c_reduce_mode == 1) { // reduce basic
        let curr_output_length = c_array_length;
        for (let i = 1; i <= c_array_length; i <<= 8) {
            const prev_output_length = curr_output_length;
            curr_output_length = Math.ceil(curr_output_length / c_reduce_kernel_segment_length);

            g_reduce_basic_kernel_dispatch_params.push(curr_output_length);

            const reduce_kernel_builder = new KernelBuilder(g_device, "reduce", reduce_shader, "compute");

            let reduce_kernel: Kernel;
            if (i == 1) { // first pass
                reduce_kernel = reduce_kernel_builder
                    .add_constant("SEGMENT_LENGTH", c_reduce_kernel_segment_length)
                    .add_buffer("array_length", 0, BufferTypeEnum.UNIFORM, g_array_length_buffer)
                    .add_buffer("input_array", 1, BufferTypeEnum.READONLY_STORAGE, g_input_array_buffer)
                    .create_then_add_buffer("output_sum_per_segment", 2, BufferTypeEnum.STORAGE, curr_output_length * 4)
                    .build();
            } else {
                const prev_kernel = g_reduce_basic_kernel_chain[g_reduce_basic_kernel_chain.length - 1];
                reduce_kernel = reduce_kernel_builder
                    .add_constant("SEGMENT_LENGTH", c_reduce_kernel_segment_length)
                    .create_then_add_buffer("array_length", 0, BufferTypeEnum.UNIFORM, 4)
                    .add_buffer("input_array", 1, BufferTypeEnum.READONLY_STORAGE, prev_kernel.get_buffer("output_sum_per_segment"))
                    .create_then_add_buffer("output_sum_per_segment", 2, BufferTypeEnum.STORAGE, curr_output_length * 4)
                    .build();

                const prev_output_length_buffer = new Uint32Array(1);
                prev_output_length_buffer[0] = prev_output_length;
                g_device.queue.writeBuffer(reduce_kernel.get_buffer("array_length"), 0, prev_output_length_buffer.buffer);
            }

            g_reduce_basic_kernel_chain.push(reduce_kernel);
        }
        console.log(`reduce basic kernel will use ${g_reduce_basic_kernel_chain.length} pass(es)`)
    } else if (c_reduce_mode == 2) { // reduce upsweep
        let curr_output_length = c_array_length;
        for (let i = 1; i <= c_array_length; i <<= 8) {
            const prev_output_length = curr_output_length;
            curr_output_length = Math.ceil(curr_output_length / c_reduce_kernel_segment_length);

            g_reduce_upsweep_kernel_dispatch_params.push(curr_output_length);

            const reduce_kernel_builder = new KernelBuilder(g_device, "reduce", reduce_upsweep_shader, "compute");

            let reduce_kernel: Kernel;
            if (i == 1) { // first pass
                reduce_kernel = reduce_kernel_builder
                    .add_constant("SEGMENT_LENGTH", c_reduce_kernel_segment_length)
                    .add_buffer("array_length", 0, BufferTypeEnum.UNIFORM, g_array_length_buffer)
                    .add_buffer("input_array", 1, BufferTypeEnum.READONLY_STORAGE, g_input_array_buffer)
                    .create_then_add_buffer("output_sum_per_segment", 2, BufferTypeEnum.STORAGE, curr_output_length * 4)
                    .build();
            } else {
                const prev_kernel = g_reduce_upsweep_kernel_chain[g_reduce_upsweep_kernel_chain.length - 1];
                reduce_kernel = reduce_kernel_builder
                    .add_constant("SEGMENT_LENGTH", c_reduce_kernel_segment_length)
                    .create_then_add_buffer("array_length", 0, BufferTypeEnum.UNIFORM, 4)
                    .add_buffer("input_array", 1, BufferTypeEnum.READONLY_STORAGE, prev_kernel.get_buffer("output_sum_per_segment"))
                    .create_then_add_buffer("output_sum_per_segment", 2, BufferTypeEnum.STORAGE, curr_output_length * 4)
                    .build();

                const prev_output_length_buffer = new Uint32Array(1);
                prev_output_length_buffer[0] = prev_output_length;
                g_device.queue.writeBuffer(reduce_kernel.get_buffer("array_length"), 0, prev_output_length_buffer.buffer);
            }

            g_reduce_upsweep_kernel_chain.push(reduce_kernel);
        }

        console.log(`reduce upsweep kernel will use ${g_reduce_upsweep_kernel_chain.length} pass(es)`);
    } else if (c_reduce_mode == 3) { // reduce flatten
        let curr_output_length = c_array_length;
        for (let i = 1; i <= c_array_length; i <<= 8) {
            const prev_output_length = curr_output_length;
            curr_output_length = Math.ceil(curr_output_length / c_reduce_kernel_segment_length);

            g_reduce_flatten_kernel_dispatch_params.push(curr_output_length);

            const reduce_kernel_builder = new KernelBuilder(g_device, "reduce", reduce_flatten_shader, "compute");

            let reduce_kernel: Kernel;
            if (i == 1) { // first pass
                reduce_kernel = reduce_kernel_builder
                    .add_constant("SEGMENT_LENGTH", c_reduce_kernel_segment_length)
                    .add_buffer("array_length", 0, BufferTypeEnum.UNIFORM, g_array_length_buffer)
                    .add_buffer("input_array", 1, BufferTypeEnum.READONLY_STORAGE, g_input_array_buffer)
                    .create_then_add_buffer("output_sum_per_segment", 2, BufferTypeEnum.STORAGE, curr_output_length * 4)
                    .build();
            } else {
                const prev_kernel = g_reduce_flatten_kernel_chain[g_reduce_flatten_kernel_chain.length - 1];
                reduce_kernel = reduce_kernel_builder
                    .add_constant("SEGMENT_LENGTH", c_reduce_kernel_segment_length)
                    .create_then_add_buffer("array_length", 0, BufferTypeEnum.UNIFORM, 4)
                    .add_buffer("input_array", 1, BufferTypeEnum.READONLY_STORAGE, prev_kernel.get_buffer("output_sum_per_segment"))
                    .create_then_add_buffer("output_sum_per_segment", 2, BufferTypeEnum.STORAGE, curr_output_length * 4)
                    .build();

                const prev_output_length_buffer = new Uint32Array(1);
                prev_output_length_buffer[0] = prev_output_length;
                g_device.queue.writeBuffer(reduce_kernel.get_buffer("array_length"), 0, prev_output_length_buffer.buffer);
            }

            g_reduce_flatten_kernel_chain.push(reduce_kernel);
        }

        console.log(`reduce flatten kernel will use ${g_reduce_flatten_kernel_chain.length} pass(es)`)
    }
}

function init_kernels_prefix_sum() {
    const prefix_sum_native_kernel_builder = new KernelBuilder(g_device, "prefix_sum_native", prefix_sum_native_shader, "compute");
    g_prefix_sum_native_kernel = prefix_sum_native_kernel_builder
        .add_buffer("array_length", 0, BufferTypeEnum.UNIFORM, g_array_length_buffer)
        .add_buffer("input_array", 1, BufferTypeEnum.READONLY_STORAGE, g_input_array_buffer)
        .create_then_add_buffer("prefix_sum", 2, BufferTypeEnum.STORAGE, c_array_length * 4)
        .build();

    const segment_cnt = Math.ceil(c_array_length / c_prefix_sum_kernel_segment_length);
    const prefix_sum_basic_kernel_builder = new KernelBuilder(g_device, "prefix_sum_basic", prefix_sum_basic_shader, "compute");
    g_prefix_sum_basic_kernel = prefix_sum_basic_kernel_builder
        .add_constant("SEGMENT_LENGTH", c_prefix_sum_kernel_segment_length)
        .add_buffer("array_length", 0, BufferTypeEnum.UNIFORM, g_array_length_buffer)
        .add_buffer("input_array", 1, BufferTypeEnum.READONLY_STORAGE, g_input_array_buffer)
        .create_then_add_buffer("prefix_sum_intermediate", 2, BufferTypeEnum.STORAGE, c_array_length * 4)
        .create_then_add_buffer("segment_sum", 3, BufferTypeEnum.STORAGE, segment_cnt * 4)
        .build();

    const prefix_sum_cum_kernel_builder = new KernelBuilder(g_device, "prefix_sum_cum", prefix_sum_cum_shader, "compute");
    g_prefix_sum_cum_kernel = prefix_sum_cum_kernel_builder
        .add_constant("SEGMENT_LENGTH", c_prefix_sum_kernel_segment_length)
        .add_buffer("array_length", 0, BufferTypeEnum.UNIFORM, g_array_length_buffer)
        .add_buffer("prefix_sum", 1, BufferTypeEnum.STORAGE, g_prefix_sum_basic_kernel.get_buffer("prefix_sum_intermediate"))
        .add_buffer("segment_sum", 2, BufferTypeEnum.READONLY_STORAGE, g_prefix_sum_basic_kernel.get_buffer("segment_sum"))
        .build();
}

function init_kernels() {
    init_kernels_prefix_sum();
    init_kernels_reduce();
}

async function compute() {
    console.warn("start computing..");

    const command_encoder = g_device.createCommandEncoder();

    function compute_reduce() {
        if (c_reduce_mode == 0) {
            g_reduce_native_kernel.dispatch(1, 1, 1, command_encoder);
        } else if (c_reduce_mode == 1) {
            g_reduce_basic_kernel_chain.forEach((reduce_kernel, index) => {
                reduce_kernel.dispatch(g_reduce_basic_kernel_dispatch_params[index], 1, 1, command_encoder);
            });
        } else if (c_reduce_mode == 2) {
            g_reduce_upsweep_kernel_chain.forEach((reduce_kernel, index) => {
                reduce_kernel.dispatch(g_reduce_upsweep_kernel_dispatch_params[index], 1, 1, command_encoder);
            });
        } else if (c_reduce_mode == 3) {
            g_reduce_flatten_kernel_chain.forEach((reduce_kernel, index) => {
                reduce_kernel.dispatch(g_reduce_flatten_kernel_dispatch_params[index], 1, 1, command_encoder);
            });
        }
    }

    function compute_prefix_sum() {
        if (c_prefix_sum_mode == 0) {
            g_prefix_sum_native_kernel.dispatch(1, 1, 1, command_encoder);
        } else if (c_prefix_sum_mode == 1) {
            const segment_cnt = Math.ceil(c_array_length / c_prefix_sum_kernel_segment_length);
            g_prefix_sum_basic_kernel.dispatch(segment_cnt, 1, 1, command_encoder);
            g_prefix_sum_cum_kernel.dispatch(segment_cnt, 1, 1, command_encoder);
        }
    }

    compute_reduce();
    compute_prefix_sum();

    g_device.queue.submit([command_encoder.finish()]);
}

async function inspect_output_reduce() {
    console.log("reference reduce result: ", g_input_array_sum);
    if (c_reduce_mode == 0) {
        console.log("reduce native --->");
        await g_reduce_native_kernel.print_buffer_uint32("output_sum");
    } else if (c_reduce_mode == 1) {
        console.log("reduce basic --->");
        await g_reduce_basic_kernel_chain[g_reduce_basic_kernel_chain.length - 1].print_buffer_uint32("output_sum_per_segment");
    } else if (c_reduce_mode == 2) {
        console.log("reduce upsweep --->");
        await g_reduce_upsweep_kernel_chain[g_reduce_upsweep_kernel_chain.length - 1].print_buffer_uint32("output_sum_per_segment");
    } else if (c_reduce_mode == 3) {
        console.log("reduce flatten --->");
        await g_reduce_flatten_kernel_chain[g_reduce_flatten_kernel_chain.length - 1].print_buffer_uint32("output_sum_per_segment");
    }
}

async function inspect_output_prefix_sum() {
    if (c_prefix_sum_mode == 0) {
        console.log("prefix sum native --->");
        await g_prefix_sum_native_kernel.print_buffer_uint32("prefix_sum");
    } else if (c_prefix_sum_mode == 1) {
        console.log("prefix sum basic --->");
        await g_prefix_sum_cum_kernel.print_buffer_uint32("prefix_sum");
        await g_prefix_sum_cum_kernel.print_buffer_uint32("segment_sum");
    }
}

async function inspect_output() {
    await inspect_output_reduce();
    await inspect_output_prefix_sum();
}

async function main() {
    await init_webgpu();
    init_input_array();
    init_kernels();
    compute();
    await inspect_output();
}

main();
