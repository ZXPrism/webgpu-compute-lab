import { Kernel } from "./kernel.ts";
import type { BufferInfo, BufferType } from "./buffer_info";
import { BufferTypeEnum } from "./buffer_info";

interface BufferEntry {
    name: string;
    binding_point: number;
    type: BufferType,
    buffer: GPUBuffer;
}

export class KernelBuilder {
    private bind_group_layout: GPUBindGroupLayout | undefined;
    private pipeline_layout: GPUPipelineLayout | undefined;

    private buffer_entries: BufferEntry[] = [];
    private constants: Map<string, number> = new Map<string, number>();

    constructor(private device: GPUDevice, public kernel_name: string, private shader_source: string, private shader_entry_point: string) {
    }

    public create_then_add_buffer(buffer_name: string, binding_point: number, type: BufferType, size_bytes: number): KernelBuilder {
        let usage = GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
        if (type === BufferTypeEnum.UNIFORM) {
            usage |= GPUBufferUsage.UNIFORM;
        } else {
            usage |= GPUBufferUsage.STORAGE;
        }

        const buffer = this.device.createBuffer({
            label: buffer_name,
            size: size_bytes,
            usage
        });

        this.buffer_entries.push({
            name: buffer_name,
            binding_point,
            type,
            buffer
        });

        return this;
    }

    public add_buffer(buffer_name: string, binding_point: number, type: BufferType, buffer: GPUBuffer): KernelBuilder {
        this.buffer_entries.push({
            name: buffer_name,
            binding_point,
            type,
            buffer
        });

        return this;
    }

    public add_constant(constant_name: string, constant: number): KernelBuilder {
        this.constants.set(constant_name, constant);

        return this;
    }

    public build(): Kernel {
        const kernel = new Kernel(this.device);

        const binding_point_set = new Set<number>();
        for (const buffer_entry of this.buffer_entries) {
            if (binding_point_set.has(buffer_entry.binding_point)) {
                console.error('duplicate binding point detected!');
            }
            binding_point_set.add(buffer_entry.binding_point);

            const buffer_info: BufferInfo = {
                buffer: buffer_entry.buffer,
            };
            kernel.map_buffer_name_to_buffer_info.set(buffer_entry.name, buffer_info);
        }

        this._init_bind_group_layout();
        this._init_pipeline_layout();
        this._init_pipeline(kernel);
        this._init_bind_group(kernel);

        return kernel;
    }

    private _init_bind_group_layout(): void {
        const bind_group_layout_entry_list: GPUBindGroupLayoutEntry[] = [];

        for (const buffer_entry of this.buffer_entries) {
            let buffer_type: GPUBufferBindingType;
            if (buffer_entry.type === BufferTypeEnum.UNIFORM) {
                buffer_type = 'uniform';
            } else if (buffer_entry.type === BufferTypeEnum.STORAGE) {
                buffer_type = 'storage';
            } else {
                buffer_type = 'read-only-storage';
            }

            bind_group_layout_entry_list.push({
                binding: buffer_entry.binding_point,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: buffer_type
                }
            });
        }

        this.bind_group_layout = this.device.createBindGroupLayout({
            label: `${this.kernel_name}BindGroupLayout`,
            entries: bind_group_layout_entry_list
        });
    }

    private _init_pipeline_layout(): void {
        this.pipeline_layout = this.device.createPipelineLayout({
            label: `${this.kernel_name}PipelineLayout`,
            bindGroupLayouts: [this.bind_group_layout]
        });
    }

    private _init_pipeline(kernel: Kernel): void {
        const shader_module = this.device.createShaderModule({
            label: `${this.kernel_name}ShaderModule`,
            code: this.shader_source,
        });

        kernel.pipeline = this.device.createComputePipeline({
            label: `${this.kernel_name}Pipeline`,
            layout: this.pipeline_layout!,
            compute: {
                module: shader_module,
                entryPoint: this.shader_entry_point,
                constants: Object.fromEntries(this.constants)
            },
        });
    }

    private _init_bind_group(kernel: Kernel): void {
        const bind_group_entry_list: GPUBindGroupEntry[] = [];
        for (const buffer_entry of this.buffer_entries) {
            bind_group_entry_list.push({
                binding: buffer_entry.binding_point,
                resource: {
                    buffer: buffer_entry.buffer
                }
            });
        }

        kernel.bind_group = this.device.createBindGroup({
            layout: kernel.pipeline!.getBindGroupLayout(0),
            entries: bind_group_entry_list,
        });
    }
}
