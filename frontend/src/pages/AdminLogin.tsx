import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Lock, User, Eye, EyeOff, ShieldCheck } from "lucide-react";
import { apiService } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";

const AdminLogin = () => {
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

            // Strict Admin Role Check
            if (user.role !== 'ADMIN') {
                toast({
                    title: "Access Denied",
                    description: "This portal is restricted to administrators only.",
                    variant: "destructive",
                });
                setIsLoading(false);
                return;
            }

            if (!user.is_active) {
                toast({
                    title: "Account Inactive",
                    description: "Your account is inactive. Please contact system administrator.",
                    variant: "destructive",
                });
                setIsLoading(false);
                return;
            }

            // Store user info
            localStorage.setItem('user', JSON.stringify(user));
            localStorage.setItem('username', username);

            // Redirect to Admin Dashboard
            navigate("/admin/dashboard");

        } catch (error: any) {
            console.error('Login error:', error);
            const errorMsg = error.message || "Could not sign in. Please try again.";

            let errorTitle = "Login Failed";
            let errorDescription = errorMsg;

            if (errorMsg.toLowerCase().includes('invalid') || errorMsg.toLowerCase().includes('not found')) {
                errorTitle = "Invalid Credentials";
                errorDescription = "Username or password is incorrect.";
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
        <div className="min-h-screen flex items-center justify-center p-4 bg-background text-foreground relative overflow-hidden">
            {/* Background Ambience - Blue Dotted for Admin */}
            <div className="absolute top-0 right-0 w-[800px] h-[800px] bg-blue-900/5 rounded-full blur-[100px] pointer-events-none -translate-y-1/2 translate-x-1/2" />
            <div className="absolute bottom-0 left-0 w-[800px] h-[800px] bg-ohg-navy/5 rounded-full blur-[100px] pointer-events-none translate-y-1/2 -translate-x-1/2" />

            <Card className="w-full max-w-md p-8 bg-white/80 backdrop-blur-xl border-blue-100 shadow-2xl rounded-3xl animate-fade-in relative z-10">
                <div className="text-center mb-10">
                    <div className="flex justify-center mb-4">
                        <div className="p-3 bg-blue-50 rounded-full border border-blue-100">
                            <ShieldCheck className="h-8 w-8 text-blue-600" />
                        </div>
                    </div>
                    <h1 className="text-3xl font-bold text-ohg-navy mb-2 tracking-tight">
                        Admin Portal
                    </h1>
                    <p className="text-muted-foreground">
                        Sign in to manage the platform
                    </p>
                </div>

                <div className="space-y-6">
                    <div className="space-y-2">
                        <label className="text-sm font-medium text-ohg-navy">
                            Username
                        </label>
                        <Input
                            type="text"
                            placeholder="Admin Username"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && handleLogin()}
                            className="bg-white border-input focus:border-blue-500 focus:ring-blue-500/20 text-foreground placeholder:text-muted-foreground h-12 shadow-sm"
                            disabled={isLoading}
                        />
                    </div>

                    <div className="space-y-2">
                        <label className="text-sm font-medium text-ohg-navy">
                            Password
                        </label>
                        <div className="relative">
                            <Input
                                type={showPassword ? "text" : "password"}
                                placeholder="Admin Password"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                onKeyDown={(e) => e.key === 'Enter' && handleLogin()}
                                className="bg-white border-input focus:border-blue-500 focus:ring-blue-500/20 text-foreground placeholder:text-muted-foreground pr-10 h-12 shadow-sm"
                                disabled={isLoading}
                            />
                            <button
                                type="button"
                                onClick={() => setShowPassword(!showPassword)}
                                className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-blue-600 transition-colors focus:outline-none"
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
                        className="w-full h-12 text-base font-bold bg-ohg-navy hover:bg-ohg-navy/90 text-white transition-all duration-300 shadow-lg disabled:opacity-50 shadow-ohg-navy/30"
                    >
                        {isLoading ? (
                            <div className="flex items-center gap-2">
                                <div className="h-4 w-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                <span>Verifying Access...</span>
                            </div>
                        ) : (
                            <><Lock className="mr-2 h-4 w-4" /> Access Dashboard</>
                        )}
                    </Button>

                    <div className="text-center mt-4">
                        <a href="/login" className="text-xs text-muted-foreground hover:text-blue-600 transition-colors font-medium">
                            Not an admin? Go to Interview Login
                        </a>
                    </div>
                </div>
            </Card>
        </div>
    );
};
export default AdminLogin;
