import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Lock, User, Eye, EyeOff, Info } from "lucide-react";
import { apiService } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";
import { supabase } from "@/lib/supabase";

const Login = () => {
  const navigate = useNavigate();
  const { toast } = useToast();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isSignUp, setIsSignUp] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const handleAuth = async () => {
    if (!email || !password) {
      toast({
        title: "Missing Credentials",
        description: "Please enter both email and password.",
        variant: "destructive",
      });
      return;
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      toast({
        title: "Invalid Email",
        description: "Please enter a valid email address.",
        variant: "destructive",
      });
      return;
    }

    setIsLoading(true);
    try {
      if (isSignUp) {
        const { error } = await supabase.auth.signUp({
          email,
          password,
        });
        if (error) throw error;

        toast({
          title: "Check your email",
          description: "We sent you a confirmation link. Please click it to verify your account.",
        });
      } else {
        const { data, error } = await supabase.auth.signInWithPassword({
          email,
          password,
        });

        if (error) throw error;

        if (data.user) {
          localStorage.setItem('user', JSON.stringify({
            ...data.user,
            username: data.user.email,
            role: 'USER'
          }));
          localStorage.setItem('username', data.user.email || '');

          try {
            if (data.user.email) {
              await apiService.createUser({
                username: data.user.email,
                email: data.user.email,
                password: password,
                role: 'USER',
                access_type: 'TRIAL',
              });
            }
          } catch (err: any) {
            const errorMsg = err.message || JSON.stringify(err);
            if (!errorMsg.toLowerCase().includes('already exists') && !errorMsg.toLowerCase().includes('unique')) {
              console.warn('Failed to sync user with backend:', err);
            }
          }

          navigate("/topic-selection");
        }
      }
    } catch (error: any) {
      console.error('Auth error:', error);
      const errorMessage = error.message || "An error occurred during authentication.";
      const isCredentialError = errorMessage.toLowerCase().includes("invalid login credentials");

      toast({
        title: isSignUp ? "Sign Up Failed" : (isCredentialError ? "Wrong Password" : "Login Failed"),
        description: isCredentialError ? "The email or password you entered is incorrect. Please try again." : errorMessage,
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4 bg-background text-foreground relative overflow-hidden">
      {/* Background Ambience */}
      <div className="absolute top-0 right-0 w-[800px] h-[800px] bg-ohg-navy/5 rounded-full blur-[100px] pointer-events-none -translate-y-1/2 translate-x-1/2" />
      <div className="absolute bottom-0 left-0 w-[800px] h-[800px] bg-ohg-orange/5 rounded-full blur-[100px] pointer-events-none translate-y-1/2 -translate-x-1/2" />

      <Card className="w-full max-w-md p-8 bg-card/80 backdrop-blur-xl border-border shadow-soft rounded-3xl animate-fade-in relative z-10 glass">
        <div className="text-center mb-10">
          <div className="flex flex-col items-center justify-center mb-6">
            <div className="relative w-20 h-20 rounded-2xl overflow-hidden shadow-md hover:scale-105 transition-transform duration-300">
              <img
                src="/ohglogo.png"
                alt="OHG Logo"
                className="w-full h-full object-cover"
              />
            </div>
          </div>
          <h1 className="text-3xl font-bold text-ohg-navy dark:text-foreground mb-2 tracking-tight">
            Welcome Back
          </h1>
          <p className="text-muted-foreground">
            {isSignUp ? "Create an account to get started" : "Sign in to access your interview dashboard"}
          </p>
        </div>

        <div className="space-y-6">
          <div className="space-y-2">
            <label className="text-sm font-medium text-ohg-navy dark:text-foreground">
              Email
            </label>
            <Input
              type="email"
              placeholder="Enter your email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleAuth()}
              className="bg-background border-input focus:border-ohg-teal focus:ring-ohg-teal/20 text-foreground placeholder:text-muted-foreground h-12 shadow-sm transition-all"
              disabled={isLoading}
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-ohg-navy dark:text-foreground">
              Password
            </label>
            <div className="relative">
              <Input
                type={showPassword ? "text" : "password"}
                placeholder="Enter your password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleAuth()}
                className="bg-background border-input focus:border-ohg-teal focus:ring-ohg-teal/20 text-foreground placeholder:text-muted-foreground pr-10 h-12 shadow-sm transition-all"
                disabled={isLoading}
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-ohg-navy dark:hover:text-foreground transition-colors focus:outline-none"
                disabled={isLoading}
              >
                {showPassword ? (
                  <EyeOff className="h-4 w-4" />
                ) : (
                  <Eye className="h-4 w-4" />
                )}
              </button>
            </div>
          </div>

          <Button
            onClick={handleAuth}
            disabled={isLoading || !email || !password}
            className="w-full h-12 text-base font-semibold bg-ohg-orange text-white hover:bg-ohg-orange-hover transition-all duration-300 shadow-lg hover:shadow-ohg-orange/30 disabled:opacity-50"
          >
            {isLoading ? (
              <div className="flex items-center gap-2">
                <div className="h-4 w-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                <span>Signing in...</span>
              </div>
            ) : (
              <><User className="mr-2 h-4 w-4" /> {isSignUp ? "Sign Up" : "Sign In"}</>
            )}
          </Button>

          <div className="text-center">
            <button
              onClick={() => setIsSignUp(!isSignUp)}
              className="text-sm text-ohg-teal hover:text-ohg-teal-light hover:underline font-medium transition-colors"
            >
              {isSignUp ? "Already have an account? Sign In" : "Don't have an account? Sign Up"}
            </button>
          </div>

          <div className="text-center p-4 bg-ohg-grey-light/50 dark:bg-muted/50 rounded-xl border border-border">
            <p className="text-sm text-muted-foreground flex items-center justify-center gap-2">
              <Info className="h-4 w-4 text-ohg-navy dark:text-foreground" />
              Contact administrator for credentials
            </p>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default Login;
