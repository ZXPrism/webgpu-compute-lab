export async function read_buffer_data<T extends ArrayBufferView>(
    device: GPUDevice,
    srcBuffer: GPUBuffer,
    size: number,
    ArrayType: { new(buffer: ArrayBufferLike): T },
    offset = 0
): Promise<T> {
    const stagingBuffer = device.createBuffer({
        size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(srcBuffer, offset, stagingBuffer, 0, size);
    device.queue.submit([encoder.finish()]);

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const copy = stagingBuffer.getMappedRange().slice(0);
    stagingBuffer.unmap();
    stagingBuffer.destroy();

    return new ArrayType(copy);
}

// from: https://github.com/kishimisu/WebGPU-Radix-Sort
// Create a timestamp query object for measuring GPU time
export function create_timestamp_query(device: GPUDevice) {
    const timestampCount = 2
    const querySet = device.createQuerySet({
        type: "timestamp",
        count: timestampCount,
    })
    const queryBuffer = device.createBuffer({
        size: 8 * timestampCount,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC
    })
    const queryResultBuffer = device.createBuffer({
        size: 8 * timestampCount,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    })

    const resolve = (encoder: GPUCommandEncoder) => {
        encoder.resolveQuerySet(querySet, 0, timestampCount, queryBuffer, 0)
        encoder.copyBufferToBuffer(queryBuffer, 0, queryResultBuffer, 0, 8 * timestampCount)
    }

    const get_timestamps = async () => {
        await queryResultBuffer.mapAsync(GPUMapMode.READ)
        const timestamps = new BigUint64Array(queryResultBuffer.getMappedRange().slice())
        queryResultBuffer.unmap();
        return timestamps
    }

    return {
        descriptor: {
            timestampWrites: {
                querySet: querySet,
                beginningOfPassWriteIndex: 0,
                endOfPassWriteIndex: 1,
            },
        },
        resolve,
        get_timestamps
    }
}
