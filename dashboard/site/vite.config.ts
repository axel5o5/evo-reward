import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { execSync } from 'child_process';

export default defineConfig({
  plugins: [
    react(),
    {
      name: 'extract-data',
      buildStart() {
        try {
          execSync('python3 ../extract_data.py', {
            cwd: __dirname,
            stdio: 'inherit',
          });
        } catch (e) {
          console.warn('Warning: extract_data.py failed. Using existing data files.');
        }
      },
    },
  ],
  base: './', // relative paths for static file serving
});
