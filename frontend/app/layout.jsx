import './globals.css';

export const metadata = {
  title: 'GBM CDSS | Clinical MRI Viewer',
  description: 'Research clinical decision-support MRI review workspace.',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
