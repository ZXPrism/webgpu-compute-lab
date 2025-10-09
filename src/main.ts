import { BufferTypeEnum } from "./buffer_info";
import type { Kernel } from "./kernel";
import { KernelBuilder } from "./kernel_builder";

import reduce_shader from "./shaders/reduce_basic.wgsl?raw";
import reduce_native_shader from "./shaders/reduce_native.wgsl?raw";
import reduce_upsweep_shader from "./shaders/reduce_upsweep.wgsl?raw";
import reduce_flatten_shader from "./shaders/reduce_flatten.wgsl?raw";

import prefix_sum_native_shader from "./shaders/prefix_sum_native.wgsl?raw";

let c_array_length = 1000000;
let c_reduce_mode = 2; // 0: native; 1: simple; 2: upsweep; 3: upsweep + no bank conflict
let c_prefix_sum_mode = 1; // 0: native

let g_device: GPUDevice;

let g_prefix_sum_native_kernel: Kernel;
let g_prefix_sum_cum_kernel: Kernel;

let g_reduce_native_kernel: Kernel;
let c_reduce_kernel_segment_length = 256;
let g_reduce_kernel_chain: Kernel[] = [];
let g_reduce_kernel_dispatch_params: number[] = [];
let g_reduce_kernel_chain_upsweep: Kernel[] = [];
let g_reduce_kernel_dispatch_params_upsweep: number[] = [];


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
    {
        let curr_output_length = c_array_length;
        for (let i = 1; i <= c_array_length; i <<= 8) {
            const prev_output_length = curr_output_length;
            curr_output_length = Math.ceil(curr_output_length / c_reduce_kernel_segment_length);

            g_reduce_kernel_dispatch_params.push(curr_output_length);

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
                const prev_kernel = g_reduce_kernel_chain[g_reduce_kernel_chain.length - 1];
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

            g_reduce_kernel_chain.push(reduce_kernel);
        }
        console.log(`reduce kernel will use ${g_reduce_kernel_chain.length} pass(es)`)
    }

    const reduce_native_kernel_builder = new KernelBuilder(g_device, "reduce_native", reduce_native_shader, "compute");
    g_reduce_native_kernel = reduce_native_kernel_builder
        .add_buffer("array_length", 0, BufferTypeEnum.UNIFORM, g_array_length_buffer)
        .add_buffer("input_array", 1, BufferTypeEnum.READONLY_STORAGE, g_input_array_buffer)
        .create_then_add_buffer("output_sum", 2, BufferTypeEnum.STORAGE, 4)
        .build();

    {
        let curr_output_length = c_array_length;
        for (let i = 1; i <= c_array_length; i <<= 8) {
            const prev_output_length = curr_output_length;
            curr_output_length = Math.ceil(curr_output_length / c_reduce_kernel_segment_length);

            g_reduce_kernel_dispatch_params_upsweep.push(curr_output_length);

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
                const prev_kernel = g_reduce_kernel_chain_upsweep[g_reduce_kernel_chain_upsweep.length - 1];
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

            g_reduce_kernel_chain_upsweep.push(reduce_kernel);
        }
        console.log(`reduce kernel will use ${g_reduce_kernel_chain_upsweep.length} pass(es)`)
    }
}

function init_kernels_prefix_sum() {
    const prefix_sum_native_kernel_builder = new KernelBuilder(g_device, "prefix_sum_native", prefix_sum_native_shader, "compute");
    g_prefix_sum_native_kernel = prefix_sum_native_kernel_builder
        .add_buffer("array_length", 0, BufferTypeEnum.UNIFORM, g_array_length_buffer)
        .add_buffer("input_array", 1, BufferTypeEnum.READONLY_STORAGE, g_input_array_buffer)
        .create_then_add_buffer("prefix_sum", 2, BufferTypeEnum.STORAGE, c_array_length * 4)
        .build();
}

function init_kernels() {
    init_kernels_prefix_sum();
    init_kernels_reduce();
}

async function compute() {
    console.warn("start computing..");

    const command_encoder = g_device.createCommandEncoder();

    function compute_reduce(mode: number) {
        if (mode == 0) {
            g_reduce_native_kernel.dispatch(1, 1, 1, command_encoder);
        } else if (mode == 1) {
            g_reduce_kernel_chain.forEach((reduce_kernel, index) => {
                reduce_kernel.dispatch(g_reduce_kernel_dispatch_params[index], 1, 1, command_encoder);
            });
        } else if (mode == 2) {
            g_reduce_kernel_chain_upsweep.forEach((reduce_kernel, index) => {
                reduce_kernel.dispatch(g_reduce_kernel_dispatch_params_upsweep[index], 1, 1, command_encoder);
            });
        } else {

        }
    }

    function compute_prefix_sum(mode: number) {
        if (mode == 0) {
            g_prefix_sum_native_kernel.dispatch(1, 1, 1, command_encoder);
        }
    }

    compute_reduce(c_reduce_mode);
    compute_prefix_sum(c_prefix_sum_mode);

    g_device.queue.submit([command_encoder.finish()]);
}

async function inspect_output_reduce(mode: number) {
    console.log("reference reduce result: ", g_input_array_sum);
    if (mode == 0) {
        console.log("reduce native --->");
        await g_reduce_native_kernel.print_buffer_uint32("output_sum");
    } else if (mode == 1) {
        console.log("reduce simple --->");
        await g_reduce_kernel_chain[g_reduce_kernel_chain.length - 1].print_buffer_uint32("output_sum_per_segment");
    } else if (mode == 2) {
        console.log("reduce upsweep --->");
        await g_reduce_kernel_chain_upsweep[g_reduce_kernel_chain.length - 1].print_buffer_uint32("output_sum_per_segment");
    } else {

    }
}

async function inspect_output_prefix_sum(mode: number) {
    if (mode == 0) {
        console.log("prefix sum native --->");
        await g_prefix_sum_native_kernel.print_buffer_uint32("prefix_sum");
    }
}

async function inspect_output() {
    await inspect_output_reduce(c_reduce_mode);
    await inspect_output_prefix_sum(c_prefix_sum_mode);
}

async function main() {
    await init_webgpu();
    init_input_array();
    init_kernels();
    compute();
    await inspect_output();
}

main();
