import { LOCALE_COOKIE_NAME } from '@/constants/common';
import { cookies as c } from 'next/headers';
import i18n from '../../../../../../i18n.json';
import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

export async function POST(req: Request) {
  const cookies = c();

  const { locale } = await req.json();

  // Check if locale is provided
  if (!locale) {
    return NextResponse.json(
      { message: 'Locale is required' },
      { status: 500 }
    );
  }

  // Check if locale is supported
  const locales = i18n.locales;

  if (!locales.includes(locale))
    return NextResponse.json(
      { message: 'Locale is not supported' },
      { status: 500 }
    );

  cookies.set(LOCALE_COOKIE_NAME, locale);
  return NextResponse.json({ message: 'Success' });
}

export async function DELETE() {
  const cookies = c();

  cookies.delete(LOCALE_COOKIE_NAME);
  return NextResponse.json({ message: 'Success' });
}
