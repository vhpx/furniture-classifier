import '../../styles/globals.css';
import { ReactNode } from 'react';

import { Analytics as VercelAnalytics } from '@vercel/analytics/react';
import { SpeedInsights as VercelInsights } from '@vercel/speed-insights/next';
import { TailwindIndicator } from '@/components/tailwind-indicator';
import { Providers } from '@/components/providers';
import { Toaster } from '@/components/ui/toaster';
import { siteConfig } from '@/constants/configs';
import NavbarPadding from './navbar-padding';
import { Metadata, Viewport } from 'next';
import { Inter } from 'next/font/google';
import { cn } from '@/lib/utils';
import Navbar from './navbar';

export const dynamic = 'force-dynamic';

interface Props {
  children: ReactNode;
  params: {
    lang: string;
  };
}

export async function generateMetadata({
  params: { lang },
}: Props): Promise<Metadata> {
  const enDescription = 'Classify your furniture with AI.';
  const viDescription = 'Phân loại nội thất của bạn bằng trí tuệ nhân tạo.';

  const description = lang === 'vi' ? viDescription : enDescription;

  return {
    title: {
      default: siteConfig.name,
      template: `%s - ${siteConfig.name}`,
    },
    metadataBase: new URL(siteConfig.url),
    description,
    keywords: [
      'Next.js',
      'React',
      'Tailwind CSS',
      'Server Components',
      'Radix UI',
    ],
    authors: [
      {
        name: 'vohoangphuc',
        url: 'https://www.vohoangphuc.com',
      },
    ],
    creator: 'vohoangphuc',
    openGraph: {
      type: 'website',
      locale: 'en_US',
      url: siteConfig.url,
      title: siteConfig.name,
      description,
      siteName: siteConfig.name,
      images: [
        {
          url: siteConfig.ogImage,
          width: 1200,
          height: 630,
          alt: siteConfig.name,
        },
      ],
    },
  };
}

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 5,
  themeColor: [
    { media: '(prefers-color-scheme: dark)', color: 'black' },
    { media: '(prefers-color-scheme: light)', color: 'white' },
  ],
  colorScheme: 'dark light',
};

const inter = Inter({ subsets: ['latin', 'vietnamese'], display: 'block' });

export async function generateStaticParams() {
  return [{ lang: 'en' }, { lang: 'vi' }];
}

export default async function RootLayout({ children, params }: Props) {
  return (
    <html lang={params.lang}>
      <body
        className={cn(
          'bg-background min-h-screen font-sans antialiased',
          inter.className
        )}
      >
        <VercelAnalytics />
        <VercelInsights />
        <Providers
          attribute="class"
          defaultTheme="dark"
          themes={[
            'system',

            'light',
            'light-pink',
            'light-purple',
            'light-yellow',
            'light-orange',
            'light-green',
            'light-blue',

            'dark',
            'dark-pink',
            'dark-purple',
            'dark-yellow',
            'dark-orange',
            'dark-green',
            'dark-blue',
          ]}
          enableColorScheme={false}
          enableSystem
        >
          <Navbar />
          <NavbarPadding>{children}</NavbarPadding>
        </Providers>
        <TailwindIndicator />
        <Toaster />
      </body>
    </html>
  );
}
