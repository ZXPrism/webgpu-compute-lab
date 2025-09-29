export const BufferTypeEnum = {
    UNIFORM: 0,
    STORAGE: 1,
    READONLY_STORAGE: 2,
} as const;

export type BufferType = number;

export interface BufferInfo {
    buffer: GPUBuffer,
}
