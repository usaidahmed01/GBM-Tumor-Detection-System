import './globals.css';

export const metadata = {
  title: 'NeuroGlioma AI | Clinical MRI Viewer',
  description: 'Neuro-oncology MRI decision-support and review workspace.',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
