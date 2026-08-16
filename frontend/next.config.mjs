/** @type {import('next').NextConfig} */
const backendOrigin = (process.env.GBM_BACKEND_ORIGIN || 'http://127.0.0.1:8000').replace(/\/$/, '');

const nextConfig = {
  reactStrictMode: true,
  async headers() {
    return [
      {
        source: '/:path*',
        headers: [
          { key: 'X-Content-Type-Options', value: 'nosniff' },
          { key: 'Referrer-Policy', value: 'same-origin' },
          { key: 'X-Frame-Options', value: 'DENY' },
          { key: 'Permissions-Policy', value: 'camera=(), microphone=(), geolocation=()' },
        ],
      },
    ];
  },
  async rewrites() {
    return [
      {
        source: '/gbm-api/:path*',
        destination: `${backendOrigin}/api/v1/:path*`,
      },
    ];
  },
  webpack: (config) => {
    // Cornerstone's compute WebWorker can make Webpack emit a runtime-chunk
    // circularity warning even when the worker bundle is valid. Keep the
    // suppression narrowly scoped to that exact runtime pair so unrelated
    // circular-dependency warnings still surface.
    config.ignoreWarnings = [
      ...(config.ignoreWarnings || []),
      (warning) => {
        const message = typeof warning === 'string' ? warning : (warning?.message || String(warning || ''));
        return /Circular dependency between chunks with runtime \(compute, webpack(?:-runtime)?\)/.test(message);
      },
    ];
    config.resolve.fallback = {
      ...(config.resolve.fallback || {}),
      fs: false,
    };
    config.module.rules.push({
      test: /\.wasm$/,
      type: 'asset/resource',
    });
    return config;
  },
};

export default nextConfig;
