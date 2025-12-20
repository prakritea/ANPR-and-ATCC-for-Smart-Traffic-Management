import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export default function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");

  return (
    <main className="min-h-screen px-6 py-24">
      <div className="mx-auto grid max-w-md gap-6">
        <h1 className="bg-gradient-to-b from-white to-white/70 bg-clip-text text-3xl font-extrabold text-transparent">Account</h1>
        <div className="rounded-2xl border border-white/10 bg-white/5 p-6">
          <Tabs defaultValue="signin">
            <TabsList className="mb-4 bg-white/5">
              <TabsTrigger value="signin">Sign In</TabsTrigger>
              <TabsTrigger value="signup">Sign Up</TabsTrigger>
            </TabsList>

            <TabsContent value="signin">
              <div className="space-y-4">
                <div>
                  <label className="mb-1 block text-sm text-white/80">Email</label>
                  <Input type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com" />
                </div>
                <div>
                  <label className="mb-1 block text-sm text-white/80">Password</label>
                  <Input type="password" value={password} onChange={(e) => setPassword(e.target.value)} />
                </div>
                <Button className="w-full rounded-full">Login</Button>
              </div>
            </TabsContent>

            <TabsContent value="signup">
              <div className="space-y-4">
                <div>
                  <label className="mb-1 block text-sm text-white/80">Full Name</label>
                  <Input value={name} onChange={(e) => setName(e.target.value)} />
                </div>
                <div>
                  <label className="mb-1 block text-sm text-white/80">Email</label>
                  <Input type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com" />
                </div>
                <div>
                  <label className="mb-1 block text-sm text-white/80">Password</label>
                  <Input type="password" value={password} onChange={(e) => setPassword(e.target.value)} />
                </div>
                <Button className="w-full rounded-full" variant="secondary">Create Account</Button>
              </div>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </main>
  );
}
