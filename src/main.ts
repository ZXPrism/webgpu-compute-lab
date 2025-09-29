import { BufferTypeEnum } from "./buffer_info";
import type { Kernel } from "./kernel";
import { KernelBuilder } from "./kernel_builder";
import prefix_sum_native_shader from "./shaders/prefix_sum_native.wgsl?raw"
import { create_timestamp_query } from "./utils"

let c_array_length = 10000;

let g_device: GPUDevice;
let g_prefix_sum_native_kernel: Kernel;

let g_array_length_buffer: GPUBuffer;
let g_input_array_buffer: GPUBuffer;

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
}

function init_kernels() {
    const prefix_sum_native_kernel_builder = new KernelBuilder(g_device, "prefix_sum_native", prefix_sum_native_shader, "compute");
    g_prefix_sum_native_kernel = prefix_sum_native_kernel_builder
        .add_buffer("array_length", 0, BufferTypeEnum.UNIFORM, g_array_length_buffer)
        .add_buffer("input_array", 1, BufferTypeEnum.READONLY_STORAGE, g_input_array_buffer)
        .create_then_add_buffer("prefix_sum", 2, BufferTypeEnum.STORAGE, c_array_length * 4)
        .build();
}

async function compute() {
    const query = create_timestamp_query(g_device);

    const command_encoder = g_device.createCommandEncoder();

    g_prefix_sum_native_kernel.dispatch(1, 1, 1, command_encoder, query.descriptor);
    query.resolve(command_encoder);

    g_device.queue.submit([command_encoder.finish()]);

    const timestamps = await query.get_timestamps();
    console.info("gpu time: ", Number(timestamps[1] - timestamps[0]) / 1e6, "ms");
}

function inspect_output() {
    g_prefix_sum_native_kernel.print_buffer_uint32("prefix_sum");
}

async function main() {
    await init_webgpu();
    init_input_array();
    init_kernels();
    await compute();
    inspect_output();
}

main();
