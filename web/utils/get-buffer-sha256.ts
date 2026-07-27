const getBufferSha256 = async (buffer: ArrayBuffer) => {
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
};

export { getBufferSha256 };
