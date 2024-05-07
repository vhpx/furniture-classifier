# Furniture Classifier Web Demo

This repository contains the web demo for the Furniture Classifier project. The project is a web application that allows users to upload an image of a piece of furniture and get a classification of the furniture type.

## What's inside?

This turborepo uses [pnpm](https://pnpm.io) as a package manager. It includes the following packages/apps:

### Apps

- `app`: a [Next.js](https://nextjs.org/) app with [Tailwind CSS](https://tailwindcss.com/) and [Shadcn UI](https://ui.shadcn.com/).

### Packages

- `eslint-config-custom`: `ESLint` configurations (includes `eslint-config-next` and `eslint-config-prettier`).
- `tsconfig`: `tsconfig.json`s used throughout the monorepo.

### Utilities

This turborepo has some additional tools already setup for you:

- [Tailwind CSS](https://tailwindcss.com/) for styles
- [Shadcn UI](https://ui.shadcn.com/) for UI components
- [TypeScript](https://www.typescriptlang.org/) for static type checking.
- [ESLint](https://eslint.org/) for code linting.
- [Prettier](https://prettier.io) for code formatting.

### Setup

Before proceeding to the Build and Develop sections, you should have pnpm installed on your local machine.
The most common way to install it is using npm:

```bash
npm install -g pnpm
```

> More information can be found at the [pnpm installation](https://pnpm.io/installation) page.

After installing pnpm, you can install all dependencies by running the following command:

```bash
pnpm install
```

or

```bash
pnpm i
```

### Build

To build all apps and packages, run the following command:

```bash
pnpm build
```

### Develop

To develop all apps and packages (without requiring a local setup), run the following command:

```bash
pnpm dev
```

To stop development apps and packages that are running on your local machine, run the following command:

```bash
pnpm stop
```

#### Better Development Experience

In case you want to run all local development servers, run the following command:

```bash
pnpm devx
```

Running `devx` will:

1. Stop the currently running supabase instance and save current data as backup (if there is any)
2. Install all dependencies
3. Start a new supabase instance (using backed up data)
4. Start all Next.js apps in development mode

If you want to have the same procedure without the backup, you can run `pnpm devrs` instead. This will:

1. Stop the currently running supabase instance (if there is any)
2. Install all dependencies
3. Start a new supabase instance (with clean data from seed.sql)
4. Start all Next.js apps in development mode

> In case you don't want to run a local supabase instance, you can run `pnpm dev` instead.
