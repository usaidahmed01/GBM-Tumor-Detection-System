/** @type {import('next').NextConfig} */
const backendOrigin = (process.env.GBM_BACKEND_ORIGIN || 'http://127.0.0.1:8000').replace(/\/$/, '');

const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: '/gbm-api/:path*',
        destination: `${backendOrigin}/api/v1/:path*`,
      },
    ];
  },
  webpack: (config) => {
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
