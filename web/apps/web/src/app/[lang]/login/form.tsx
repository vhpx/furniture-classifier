'use client';

import { zodResolver } from '@hookform/resolvers/zod';
import { useForm } from 'react-hook-form';
import * as z from 'zod';

import { Button } from '@/components/ui/button';
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from '@/components/ui/form';
import { Input } from '@/components/ui/input';
import { toast } from '@/components/ui/use-toast';
import { useEffect, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import useTranslation from 'next-translate/useTranslation';
import { DEV_MODE } from '@/constants/common';
import Link from 'next/link';
import { IconBrandGmail, IconBrandWindows } from '@tabler/icons-react';
import { Mail } from 'lucide-react';

const FormSchema = z.object({
  email: z
    .string()
    .email()
    .refine(
      (email) => email.endsWith('rmit.edu.vn') || email.endsWith('rmit.edu.au')
    ),
  otp: z.string(),
});

export default function LoginForm() {
  const { t } = useTranslation('login');
  const router = useRouter();
  const searchParams = useSearchParams();

  const form = useForm<z.infer<typeof FormSchema>>({
    resolver: zodResolver(FormSchema),
    defaultValues: {
      email: '',
      otp: '',
    },
  });

  const [otpSent, setOtpSent] = useState(false);
  const [loading, setLoading] = useState(false);

  // Resend cooldown
  const cooldown = 60;
  const [resendCooldown, setResendCooldown] = useState(0);

  // Update resend cooldown OTP is sent
  useEffect(() => {
    if (otpSent) setResendCooldown(cooldown);
  }, [otpSent]);

  // Reduce cooldown by 1 every second
  useEffect(() => {
    if (resendCooldown > 0) {
      const timer = setTimeout(() => {
        setResendCooldown((prev) => prev - 1);
      }, 1000);

      return () => clearTimeout(timer);
    }
  }, [resendCooldown]);

  const sendOtp = async (data: { email: string }) => {
    setLoading(true);

    const res = await fetch('/api/auth/otp/send', {
      method: 'POST',
      body: JSON.stringify(data),
    });

    if (res.ok) {
      // Notify user
      toast({
        title: t('success'),
        description: t('otp_sent'),
      });

      // OTP has been sent
      setOtpSent(true);
    } else {
      toast({
        title: t('failed'),
        description: t('failed_to_send'),
      });
    }

    setLoading(false);
  };

  const verifyOtp = async (data: { email: string; otp: string }) => {
    setLoading(true);

    const res = await fetch('/api/auth/otp/verify', {
      method: 'POST',
      body: JSON.stringify(data),
    });

    if (res.ok) {
      const nextUrl = searchParams.get('nextUrl');
      router.push(nextUrl ?? '/classify');
      router.refresh();
    } else {
      setLoading(false);
      toast({
        title: t('failed'),
        description: t('failed_to_verify'),
      });
    }
  };

  async function onSubmit(data: z.infer<typeof FormSchema>) {
    const { email, otp } = data;

    if (!otpSent) await sendOtp({ email });
    else if (otp) await verifyOtp({ email, otp });
    else {
      setLoading(false);
      toast({
        title: 'Error',
        description:
          'Please enter the OTP code sent to your email to continue.',
      });
    }
  }

  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-3">
        <FormField
          control={form.control}
          name="email"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Email</FormLabel>
              <FormControl>
                <Input
                  placeholder={t('email_placeholder')}
                  {...field}
                  disabled={otpSent || loading}
                />
              </FormControl>

              {otpSent || (
                <FormDescription>{t('email_description')}</FormDescription>
              )}
              <FormMessage />
            </FormItem>
          )}
        />

        <FormField
          control={form.control}
          name="otp"
          render={({ field }) => (
            <FormItem className={otpSent ? '' : 'hidden'}>
              <FormLabel>{t('otp_code')}</FormLabel>
              <FormControl>
                <div className="flex flex-col gap-2 md:flex-row">
                  <Input placeholder="••••••" {...field} disabled={loading} />
                  <Button
                    onClick={() => sendOtp({ email: form.getValues('email') })}
                    disabled={loading || resendCooldown > 0}
                    className="md:w-40"
                    variant="secondary"
                    type="button"
                  >
                    {resendCooldown > 0
                      ? `${t('resend')} (${resendCooldown})`
                      : t('resend')}
                  </Button>
                </div>
              </FormControl>
              <FormDescription>{t('otp_description')}</FormDescription>
              <FormMessage />
            </FormItem>
          )}
        />

        {otpSent && (
          <div className="grid gap-2 md:grid-cols-2">
            {DEV_MODE ? (
              <Link
                href="http://localhost:8004/monitor"
                target="_blank"
                className="col-span-full"
              >
                <Button type="button" className="w-full" variant="outline">
                  <Mail size={18} className="mr-1" />
                  {t('open_inbucket')}
                </Button>
              </Link>
            ) : (
              <>
                <Link
                  href="https://mail.google.com/mail/u/0/#inbox"
                  target="_blank"
                >
                  <Button type="button" className="w-full" variant="outline">
                    <IconBrandGmail size={18} className="mr-1" />
                    {t('open_gmail')}
                  </Button>
                </Link>

                <Link
                  href="https://outlook.live.com/mail/inbox"
                  target="_blank"
                >
                  <Button type="button" className="w-full" variant="outline">
                    <IconBrandWindows size={18} className="mr-1" />
                    {t('open_outlook')}
                  </Button>
                </Link>
              </>
            )}
          </div>
        )}

        <div className="grid gap-2">
          <Button
            type="submit"
            className="w-full"
            disabled={
              loading ||
              form.formState.isSubmitting ||
              !form.formState.isValid ||
              (otpSent && !form.formState.dirtyFields.otp)
            }
          >
            {loading ? t('processing') : t('continue')}
          </Button>
        </div>
      </form>
    </Form>
  );
}
