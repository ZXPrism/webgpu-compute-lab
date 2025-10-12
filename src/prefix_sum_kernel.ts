import { BufferTypeEnum } from "./buffer_info";
import { c_array_length, c_prefix_sum_kernel_segment_length } from "./config";
import type { Kernel } from "./kernel";
import { KernelBuilder } from "./kernel_builder";

export class PrefixSumKernel {
    private _device: GPUDevice;
    private _array_length_buffer: GPUBuffer;
    private _input_array_buffer: GPUBuffer;

    private _prefix_sum_shader: string;
    private _prefix_sum_cum_shader: string;

    private _prefix_sum_kernel_list: Kernel[] = [];
    private _cum_kernel_list: Kernel[] = [];
    private _dispatch_params: number[] = [];

    constructor(device: GPUDevice, prefix_sum_shader: string, prefix_sum_cum_shader: string,
        array_length_buffer: GPUBuffer, input_array_buffer: GPUBuffer
    ) {
        this._device = device;
        this._array_length_buffer = array_length_buffer;
        this._input_array_buffer = input_array_buffer;

        this._prefix_sum_shader = prefix_sum_shader;
        this._prefix_sum_cum_shader = prefix_sum_cum_shader;

        this._create_prefix_sum_kernels_recursively();
    }

    public dispatch(command_encoder: GPUCommandEncoder) {
        this._prefix_sum_kernel_list.forEach((kernel, index) => {
            kernel.dispatch(this._dispatch_params[index], 1, 1, command_encoder);
        });
        this._cum_kernel_list.forEach((kernel, index) => {
            kernel.dispatch(this._dispatch_params.at(-index - 2)!, 1, 1, command_encoder);
        });
    }

    public async inspect() {
        if (this._cum_kernel_list.length == 0) {
            await this._prefix_sum_kernel_list[0].print_buffer_uint32("prefix_sum");
        } else {
            await this._cum_kernel_list.at(-1)?.print_buffer_uint32("prefix_sum");
        }
    }

    private _create_prefix_sum_kernels_recursively() {
        const prefix_sum_basic_kernel_builder = new KernelBuilder(this._device, "prefix_sum_kernel", this._prefix_sum_shader, "compute");

        let kernel: Kernel;
        let segment_cnt: number;

        if (this._prefix_sum_kernel_list.length == 0) {
            segment_cnt = Math.ceil(c_array_length / c_prefix_sum_kernel_segment_length);
            this._dispatch_params.push(segment_cnt);

            kernel = prefix_sum_basic_kernel_builder
                .add_constant("SEGMENT_LENGTH", c_prefix_sum_kernel_segment_length)
                .add_buffer("array_length", 0, BufferTypeEnum.UNIFORM, this._array_length_buffer)
                .add_buffer("input_array", 1, BufferTypeEnum.READONLY_STORAGE, this._input_array_buffer)
                .create_then_add_buffer("prefix_sum", 2, BufferTypeEnum.STORAGE, c_array_length * 4)
                .create_then_add_buffer("segment_sum", 3, BufferTypeEnum.STORAGE, segment_cnt * 4)
                .build();
        } else {
            const last_idx = this._prefix_sum_kernel_list.length - 1;
            const prev_output_length = this._dispatch_params[last_idx];

            segment_cnt = Math.ceil(prev_output_length / c_prefix_sum_kernel_segment_length);
            this._dispatch_params.push(segment_cnt);

            kernel = prefix_sum_basic_kernel_builder
                .add_constant("SEGMENT_LENGTH", c_prefix_sum_kernel_segment_length)
                .create_then_add_buffer("array_length", 0, BufferTypeEnum.UNIFORM, 4)
                .add_buffer("input_array", 1, BufferTypeEnum.READONLY_STORAGE, this._prefix_sum_kernel_list[last_idx].get_buffer("segment_sum"))
                .create_then_add_buffer("prefix_sum", 2, BufferTypeEnum.STORAGE, prev_output_length * 4)
                .create_then_add_buffer("segment_sum", 3, BufferTypeEnum.STORAGE, segment_cnt * 4)
                .build();

            const prev_output_length_buffer = new Uint32Array(1);
            prev_output_length_buffer[0] = prev_output_length;
            this._device.queue.writeBuffer(kernel.get_buffer("array_length"), 0, prev_output_length_buffer.buffer);
        }
        this._prefix_sum_kernel_list.push(kernel);
        console.log(`created prefix sum kernel with dispatch param ${segment_cnt}`);


        if (segment_cnt == 1) {
            return kernel;
        }

        const next_kernel: Kernel = this._create_prefix_sum_kernels_recursively();

        const prefix_sum_cum_kernel_builder = new KernelBuilder(this._device, "prefix_sum_cum_kernel", this._prefix_sum_cum_shader, "compute");
        const cum_kernel: Kernel = prefix_sum_cum_kernel_builder
            .add_constant("SEGMENT_LENGTH", c_prefix_sum_kernel_segment_length)
            .add_buffer("array_length", 0, BufferTypeEnum.UNIFORM, kernel.get_buffer("array_length"))
            .add_buffer("prefix_sum", 1, BufferTypeEnum.STORAGE, kernel.get_buffer("prefix_sum"))
            .add_buffer("segment_prefix_sum", 2, BufferTypeEnum.READONLY_STORAGE, next_kernel.get_buffer("prefix_sum"))
            .build();
        this._cum_kernel_list.push(cum_kernel);

        console.log(`created prefix sum CUM kernel with array length ${kernel.get_buffer("prefix_sum").size / 4}`);

        return cum_kernel;
    }
}
