const readResponseBuffer = async (
  response: Response,
  expectedByteCount: number,
  onProgress?: (loadedByteCount: number, totalByteCount: number) => void,
) => {
  if (!response.body) {
    const buffer = await response.arrayBuffer();
    onProgress?.(buffer.byteLength, expectedByteCount);
    return buffer;
  }

  const reader = response.body.getReader();
  const bytes = new Uint8Array(expectedByteCount);
  let loadedByteCount = 0;

  while (true) {
    const result = await reader.read();

    if (result.done) {
      break;
    }

    if (loadedByteCount + result.value.byteLength > expectedByteCount) {
      throw new Error("Go model response exceeded its declared size.");
    }

    bytes.set(result.value, loadedByteCount);
    loadedByteCount += result.value.byteLength;
    onProgress?.(loadedByteCount, expectedByteCount);
  }

  if (loadedByteCount !== expectedByteCount) {
    throw new Error("Go model response was incomplete.");
  }

  return bytes.buffer;
};

export { readResponseBuffer };
