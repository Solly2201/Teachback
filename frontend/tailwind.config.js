/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        brand: {
          DEFAULT: '#A5231B',
          dark: '#8C1D16',
          darker: '#701710',
          light: '#FBEAE9',
        },
        charcoal: {
          DEFAULT: '#3F3F46',
          light: '#666666',
        },
      },
      fontFamily: {
        sans: ['"Segoe UI"', 'system-ui', '-apple-system', 'Roboto', 'Arial', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
