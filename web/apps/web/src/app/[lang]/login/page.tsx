import LoginForm from './form';

export default async function Login() {
  return (
    <div className="absolute left-1/2 top-1/2 flex h-full w-full -translate-x-1/2 -translate-y-1/2 transform flex-col items-center justify-center p-8">
      <div className="grid gap-2 sm:max-w-md">
        <div className="flex items-center justify-center">
          <h1 className="relative mb-4 text-center font-mono text-4xl font-bold lg:text-7xl">
            Furniture
            <br />
            Classifier
          </h1>
        </div>

        <LoginForm />
      </div>
    </div>
  );
}
