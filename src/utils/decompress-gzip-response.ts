const decompressGzipResponse = (response: Response) => {
  if (!response.body) {
    throw new Error("Compressed Go model response has no readable body.");
  }

  return new Response(response.body.pipeThrough(new DecompressionStream("gzip")));
};

export { decompressGzipResponse };
