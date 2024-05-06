import { DEV_MODE } from './common';

export const siteConfig = {
  name: 'Furniture Classifier',
  url: DEV_MODE ? 'http://localhost:9999' : 'https://rmit-fc.vercel.app',
  ogImage: DEV_MODE
    ? 'http://localhost:9999/media/logos/og-image.png'
    : 'https://rmit-fc.vercel.app/media/logos/og-image.png',
};

export type SiteConfig = typeof siteConfig;
