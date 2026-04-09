import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { execSync } from 'child_process';
import { cpSync, existsSync, mkdirSync } from 'fs';
import { resolve } from 'path';

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
    {
      name: 'copy-docs',
      buildStart() {
        // Copy docs/*.md into public/docs/ so they're served as static files
        const docsSource = resolve(__dirname, '../../docs');
        const docsDest = resolve(__dirname, 'public/docs');
        if (existsSync(docsSource)) {
          mkdirSync(docsDest, { recursive: true });
          cpSync(docsSource, docsDest, {
            recursive: true,
            filter: (src) => {
              // Copy .md files and directories, skip PDFs and other large files
              if (src.endsWith('.pdf')) return false;
              return true;
            },
          });
          console.log('Copied docs/ -> public/docs/');
        }
      },
    },
  ],
  base: './', // relative paths for static file serving
});
