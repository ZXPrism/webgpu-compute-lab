import type { BufferInfo } from "./buffer_info";
import { read_buffer_data } from "./utils";

export class Kernel {
    public pipeline: GPUComputePipeline | undefined;
    public bind_group: GPUBindGroup | undefined;
    public map_buffer_name_to_buffer_info: Map<string, BufferInfo>;

    constructor(private device: GPUDevice) {
        this.map_buffer_name_to_buffer_info = new Map<string, BufferInfo>();
    }

    public dispatch(wg_dim_x: number, wg_dim_y: number, wg_dim_z: number, encoder: GPUCommandEncoder, descriptor?: GPUComputePassDescriptor): void {
        const compute_pass_encoder = encoder.beginComputePass(descriptor);
        compute_pass_encoder.setPipeline(this.pipeline!);
        compute_pass_encoder.setBindGroup(0, this.bind_group);
        compute_pass_encoder.dispatchWorkgroups(wg_dim_x, wg_dim_y, wg_dim_z);
        compute_pass_encoder.end();
    }

    public get_buffer(name: string): GPUBuffer {
        return this.map_buffer_name_to_buffer_info.get(name)?.buffer!;
    }

    public async print_buffer_custom(buffer_name: string, callback: (name: string, u8arr: ArrayBufferLike) => void) {
        const buffer = this.map_buffer_name_to_buffer_info.get(buffer_name)!.buffer;
        const raw = await read_buffer_data<Uint8Array>(this.device, buffer, buffer.size, Uint8Array);
        callback(buffer_name, raw.buffer);
    }

    public async print_buffer_uint32(buffer_name: string) {
        const buffer = this.map_buffer_name_to_buffer_info.get(buffer_name)!.buffer;
        const raw = await read_buffer_data<Uint32Array>(this.device, buffer, buffer.size, Uint32Array);
        console.log(`[${buffer_name}]: `, raw);
    }
}
