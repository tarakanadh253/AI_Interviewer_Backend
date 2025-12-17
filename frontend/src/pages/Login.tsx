import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Lock, User, Eye, EyeOff } from "lucide-react";
import { apiService } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";

const Login = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const handleLogin = async () => {
    if (!username || !password) {
      toast({
        title: "Missing Credentials",
        description: "Please enter both username and password.",
        variant: "destructive",
      });
      return;
    }

    setIsLoading(true);
    try {
      // Login with username and password
      const user = await apiService.login(username, password);
      console.log('Login successful:', user);

      if (!user.is_active) {
        toast({
          title: "Account Inactive",
          description: "Your account is inactive. Please contact administrator.",
          variant: "destructive",
        });
        setIsLoading(false);
        return;
      }

      // Store user info in localStorage immediately
      localStorage.setItem('user', JSON.stringify(user));
      localStorage.setItem('username', username);

      // Check for Admin Role
      if (user.role === 'ADMIN') {
        navigate("/admin");
        return;
      }

      // Check trial eligibility for regular users
      const trialCheck = await apiService.checkTrial(username);

      if (!trialCheck.can_start_interview) {
        if (trialCheck.access_type === 'TRIAL') {
          toast({
            title: "Trial Already Used",
            description: "You have already used your free trial interview. Please contact administrator for full access.",
            variant: "destructive",
          });
        } else {
          toast({
            title: "Cannot Start Interview",
            description: "Unable to start interview. Please contact administrator.",
            variant: "destructive",
          });
        }
        setIsLoading(false);
        return;
      }

      navigate("/topic-selection");
    } catch (error: any) {
      console.error('Login error:', error);
      const errorMsg = error.message || "Could not sign in. Please try again.";

      let errorTitle = "Login Failed";
      let errorDescription = errorMsg;

      if (errorMsg.toLowerCase().includes('invalid') || errorMsg.toLowerCase().includes('not found')) {
        errorTitle = "Invalid Credentials";
        errorDescription = "Username or password is incorrect. Please check your credentials or contact administrator.";
      } else if (errorMsg.toLowerCase().includes('inactive')) {
        errorTitle = "Account Inactive";
        errorDescription = "Your account is inactive. Please contact administrator.";
      }

      toast({
        title: errorTitle,
        description: errorDescription,
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (

    <div className="min-h-screen flex items-center justify-center p-4 bg-slate-950 text-slate-200 relative overflow-hidden">
      {/* Background Ambience */}
      <div className="absolute top-0 right-0 w-[600px] h-[600px] bg-blue-900/10 rounded-full blur-[100px] pointer-events-none" />
      <div className="absolute bottom-0 left-0 w-[600px] h-[600px] bg-indigo-900/10 rounded-full blur-[100px] pointer-events-none" />

      <Card className="w-full max-w-md p-8 bg-slate-900/50 backdrop-blur-xl border-slate-800 shadow-2xl rounded-3xl animate-fade-in relative z-10">
        <div className="text-center mb-10">
          <h1 className="text-3xl font-bold text-white mb-2 tracking-tight">
            Welcome Back
          </h1>
          <p className="text-slate-400">
            Sign in to access your interview dashboard
          </p>
        </div>

        <div className="space-y-6">
          <div className="space-y-2">
            <label className="text-sm font-medium text-slate-300">
              Username
            </label>
            <Input
              type="text"
              placeholder="Enter your username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleLogin()}
              className="bg-slate-950/50 border-slate-700 focus:border-blue-500 focus:ring-blue-500/20 text-white placeholder:text-slate-600 h-12"
              disabled={isLoading}
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-slate-300">
              Password
            </label>
            <div className="relative">
              <Input
                type={showPassword ? "text" : "password"}
                placeholder="Enter your password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleLogin()}
                className="bg-slate-950/50 border-slate-700 focus:border-blue-500 focus:ring-blue-500/20 text-white placeholder:text-slate-600 pr-10 h-12"
                disabled={isLoading}
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300 transition-colors focus:outline-none"
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
            onClick={handleLogin}
            disabled={isLoading || !username || !password}
            className="w-full h-12 text-base font-semibold bg-white text-slate-950 hover:bg-slate-200 transition-all duration-300 shadow-lg disabled:opacity-50"
          >
            {isLoading ? (
              <div className="flex items-center gap-2">
                <div className="h-4 w-4 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
                <span>Signing in...</span>
              </div>
            ) : (
              <><User className="mr-2 h-4 w-4" /> Sign In</>
            )}
          </Button>

          <div className="text-center p-4 bg-slate-950/30 rounded-xl border border-slate-800/50">
            <p className="text-sm text-slate-500 flex items-center justify-center gap-2">
              <span>ℹ️</span>
              Contact administrator for credentials
            </p>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default Login;
