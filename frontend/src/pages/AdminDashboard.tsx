import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "@/lib/supabase";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Plus, Edit, Trash2, Users, FileText, Eye, EyeOff, RefreshCw, ExternalLink, BookOpen, Lock, ChevronDown, ChevronUp, ArrowLeft, LayoutList, LayoutGrid } from "lucide-react";
import { apiService, type UserProfile, type Course, type Round, type Question, type InterviewSession } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";

// The UserProfile and InterviewSession interfaces are now imported from "@/lib/api"
// and are no longer defined locally.

const AdminDashboard = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState("users");
  const [users, setUsers] = useState<UserProfile[]>([]);
  const [viewMode, setViewMode] = useState<"list" | "grid">("list");

  // Security Check
  useEffect(() => {
    const checkAuth = () => {
      const userStr = localStorage.getItem('user');
      if (!userStr) {
        navigate("/admin");
        return;
      }
      try {
        const user = JSON.parse(userStr);
        if (user.role !== 'ADMIN') {
          navigate("/admin");
          return;
        }
      } catch (e) {
        navigate("/admin");
      }
    };
    checkAuth();
  }, [navigate]);

  const handleLogout = () => {
    localStorage.removeItem('user');
    localStorage.removeItem('username');
    navigate('/admin');
  };

  const [showQuestionForm, setShowQuestionForm] = useState(false);
  const [showUserForm, setShowUserForm] = useState(false);
  const [isLoadingUsers, setIsLoadingUsers] = useState(false);

  // User form state
  const [userFormData, setUserFormData] = useState({
    username: "",
    password: "",
    email: "",
    name: "",
    is_active: true,
    role: 'USER' as 'ADMIN' | 'USER',
    access_type: 'TRIAL' as 'TRIAL' | 'FULL',
  });
  const [showPassword, setShowPassword] = useState(false);
  const [isCreatingUser, setIsCreatingUser] = useState(false);

  // User details dialog state
  const [selectedUser, setSelectedUser] = useState<UserProfile | null>(null);
  const [showUserDetails, setShowUserDetails] = useState(false);
  const [userSessions, setUserSessions] = useState<InterviewSession[]>([]);
  const [isLoadingSessions, setIsLoadingSessions] = useState(false);
  const [selectedSessionId, setSelectedSessionId] = useState<number | null>(null);
  const [selectedSessionDetails, setSelectedSessionDetails] = useState<InterviewSession | null>(null);
  const [isLoadingSessionDetails, setIsLoadingSessionDetails] = useState(false);
  const [newPassword, setNewPassword] = useState("");
  const [showPasswordChange, setShowPasswordChange] = useState(false);

  // Questions state
  const [questions, setQuestions] = useState<Question[]>([]);
  const [topics, setTopics] = useState<Course[]>([]); // Changed from Topic[] to Course[]
  const [isLoadingQuestions, setIsLoadingQuestions] = useState(false);
  const [questionFormData, setQuestionFormData] = useState({
    source_type: "MANUAL" as "MANUAL" | "LINK",
    topic: "",
    round: "",
    question_text: "",
    ideal_answer: "",
    difficulty: "MEDIUM" as "EASY" | "MEDIUM" | "HARD",
    reference_links: "",
    is_active: true,
  });
  const [editingQuestion, setEditingQuestion] = useState<Question | null>(null);

  const [selectedCourseFilter, setSelectedCourseFilter] = useState<string>("ALL");

  // Topics state
  const [isLoadingTopics, setIsLoadingTopics] = useState(false);
  const [expandedCourseId, setExpandedCourseId] = useState<number | null>(null);

  const [showTopicForm, setShowTopicForm] = useState(false);
  const [topicFormData, setTopicFormData] = useState({
    name: "",
    description: "",
  });
  const [editingTopic, setEditingTopic] = useState<Course | null>(null); // Changed from Topic | null to Course | null

  // Rounds State
  const [selectedLevel, setSelectedLevel] = useState<string | null>(null);
  const [selectedRound, setSelectedRound] = useState<Round | null>(null);
  const [rounds, setRounds] = useState<Round[]>([]);
  const [isLoadingRounds, setIsLoadingRounds] = useState(false);
  const [showRoundForm, setShowRoundForm] = useState(false);
  const [editingRound, setEditingRound] = useState<Round | null>(null);
  const [roundFormData, setRoundFormData] = useState({
    name: "",
    level: "BEGINNER" as "BEGINNER" | "INTERMEDIATE" | "ADVANCED",
  });

  // Default view mode to grid for topics
  useEffect(() => {
    setViewMode("grid");
  }, []);

  // Fetch users on component mount
  useEffect(() => {
    fetchUsers();
    fetchQuestions();
    fetchTopics();
  }, []);

  // Fetch user sessions when viewing details
  useEffect(() => {
    if (selectedUser && showUserDetails) {
      fetchUserSessions(selectedUser.username);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedUser, showUserDetails]);

  const fetchUsers = async () => {
    setIsLoadingUsers(true);
    try {
      const fetchedUsers = await apiService.getUsers();
      // Ensure we always have an array
      if (Array.isArray(fetchedUsers)) {
        setUsers(fetchedUsers);
      } else {
        console.error('Invalid users response format:', fetchedUsers);
        setUsers([]);
        toast({
          title: "Error",
          description: "Invalid response format from server.",
          variant: "destructive",
        });
      }
    } catch (error: any) {
      console.error('Error fetching users:', error);
      setUsers([]); // Ensure users is always an array
      toast({
        title: "Error",
        description: error.message || "Failed to load users. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsLoadingUsers(false);
    }
  };

  const handleViewUserDetails = (user: UserProfile) => {
    setSelectedUser(user);
    setShowUserDetails(true);
  };

  const fetchUserSessions = async (username: string) => {
    setIsLoadingSessions(true);
    try {
      const sessions = await apiService.getSessionsByUsername(username);
      // Ensure we always have an array
      if (Array.isArray(sessions)) {
        setUserSessions(sessions);
      } else {
        console.error('Invalid sessions response format:', sessions);
        setUserSessions([]);
      }
    } catch (error: any) {
      console.error('Error fetching user sessions:', error);
      setUserSessions([]);
      toast({
        title: "Error",
        description: error.message || "Failed to load user sessions.",
        variant: "destructive",
      });
    } finally {
      setIsLoadingSessions(false);
    }
  };

  const handleViewSessionDetails = async (sessionId: number) => {
    if (selectedSessionId === sessionId) {
      // Toggle off
      setSelectedSessionId(null);
      setSelectedSessionDetails(null);
      return;
    }

    setSelectedSessionId(sessionId);
    setIsLoadingSessionDetails(true);
    try {
      const details = await apiService.getSessionResults(sessionId);
      setSelectedSessionDetails(details);
    } catch (error: any) {
      console.error('Error fetching session details:', error);
      toast({
        title: "Error",
        description: "Failed to load session details",
        variant: "destructive",
      });
    } finally {
      setIsLoadingSessionDetails(false);
    }
  };

  const handleResetTrial = async (userId: number) => {
    if (!confirm("Are you sure you want to reset the trial status for this user?")) return;

    try {
      await apiService.resetUserTrial(userId);
      toast({
        title: "Success",
        description: "Trial status has been reset successfully",
      });
      fetchUsers();
      if (selectedUser && selectedUser.id === userId) {
        // Refresh selected user data
        const updatedUsers = await apiService.getUsers();
        const updatedUser = updatedUsers.find(u => u.id === userId);
        if (updatedUser) {
          setSelectedUser(updatedUser);
        }
      }
    } catch (error: any) {
      toast({
        title: "Error",
        description: error.message || "Failed to reset trial status",
        variant: "destructive",
      });
    }
  };

  const handleToggleUserStatus = async (user: UserProfile) => {
    try {
      await apiService.toggleUserStatus(user.id);
      toast({
        title: "Success",
        description: `User ${user.is_active ? 'deactivated' : 'activated'} successfully`,
      });
      fetchUsers();
      if (selectedUser && selectedUser.id === user.id) {
        // Refresh selected user data
        const updatedUsers = await apiService.getUsers();
        const updatedUser = updatedUsers.find(u => u.id === user.id);
        if (updatedUser) {
          setSelectedUser(updatedUser);
        }
      }
    } catch (error: any) {
      toast({
        title: "Error",
        description: error.message || "Failed to update user status",
        variant: "destructive",
      });
    }
  };

  const handleDeleteUser = async (userId: number) => {
    if (!confirm("Are you sure you want to delete this user? This action cannot be undone.")) return;

    try {
      await apiService.deleteAdminUser(userId);
      toast({
        title: "Success",
        description: "User deleted successfully",
      });
      fetchUsers();
      if (selectedUser && selectedUser.id === userId) {
        setShowUserDetails(false);
        setSelectedUser(null);
      }
    } catch (error: any) {
      toast({
        title: "Error",
        description: error.message || "Failed to delete user",
        variant: "destructive",
      });
    }
  };

  const handleUpdateAccessType = async () => {
    if (!selectedUser) return;

    const newAccessType = selectedUser.access_type === 'FULL' ? 'TRIAL' : 'FULL';

    try {
      await apiService.updateAdminUser(selectedUser.id, {
        access_type: newAccessType
      });

      setSelectedUser({ ...selectedUser, access_type: newAccessType });
      toast({
        title: "Success",
        description: `User access updated to ${newAccessType === 'FULL' ? 'Full Access' : 'Trial'}`,
      });
      fetchUsers();
    } catch (error: any) {
      toast({
        title: "Error",
        description: error.message || "Failed to update access type",
        variant: "destructive",
      });
    }
  };

  const handleChangePassword = async () => {
    if (!selectedUser || !newPassword) return;

    try {
      await apiService.updateAdminUser(selectedUser.id, {
        password: newPassword
      });

      toast({
        title: "Success",
        description: "Password updated successfully",
      });
      setNewPassword("");
      setShowPasswordChange(false);
    } catch (error: any) {
      toast({
        title: "Error",
        description: error.message || "Failed to update password",
        variant: "destructive",
      });
    }
  };

  const handleCreateUser = async () => {
    if (!userFormData.username || !userFormData.password || !userFormData.email) {
      toast({
        title: "Missing Fields",
        description: "Please fill in username, password, and email.",
        variant: "destructive",
      });
      return;
    }

    setIsCreatingUser(true);
    try {
      // 1. Create User in Supabase Auth (Sends Confirmation Email)
      const { data: authData, error: authError } = await supabase.auth.signUp({
        email: userFormData.email,
        password: userFormData.password,
        options: {
          data: {
            username: userFormData.username,
            full_name: userFormData.name || '',
          }
        }
      });

      if (authError) {
        console.error("Supabase Auth Error:", authError);
        toast({
          title: "Auth Creation Warning",
          description: `Supabase Error: ${authError.message}. Proceeding to create in local DB...`,
          variant: "destructive",
        });
      } else {
        toast({
          title: "Confirmation Email Sent",
          description: `A confirmation email has been sent to ${userFormData.email}.`,
        });
      }

      // 2. Create User in Backend Database
      await apiService.createUser({
        username: userFormData.username,
        password: userFormData.password,
        email: userFormData.email,
        name: userFormData.name || undefined,
        is_active: userFormData.is_active,
        role: userFormData.role,
        access_type: userFormData.role === 'USER' ? userFormData.access_type : undefined,
      });

      toast({
        title: "User Created",
        description: `User "${userFormData.username}" has been created successfully.`,
        variant: "success",
      });

      // Reset form
      setUserFormData({
        username: "",
        password: "",
        email: "",
        name: "",
        is_active: true,
        role: 'USER',
        access_type: 'TRIAL',
      });
      setShowUserForm(false);

      // Refresh users list
      fetchUsers();
    } catch (error: any) {
      console.error('Error creating user:', error);
      toast({
        title: "Error",
        description: error.message || "Failed to create user. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsCreatingUser(false);
    }
  };

  const fetchQuestions = async () => {
    // Only fetch if we have a selected round or at least a course
    if (!expandedCourseId) return;

    setIsLoadingQuestions(true);
    try {
      // If a round is selected, filter by it. Otherwise filter by course.
      const roundId = selectedRound ? selectedRound.id : undefined;
      const data = await apiService.getAdminQuestions(expandedCourseId, roundId);
      setQuestions(data);
    } catch (error) {
      console.error("Error fetching questions:", error);
      toast({
        title: "Error",
        description: (error as Error).message || "Failed to fetch questions",
        variant: "destructive",
      });
    } finally {
      setIsLoadingQuestions(false);
    }
  };

  const fetchRounds = async (topicId: number, level: string) => {
    setIsLoadingRounds(true);
    try {
      const data = await apiService.getAdminRounds(topicId, level);
      setRounds(data);
    } catch (error) {
      console.error("Error fetching rounds:", error);
      toast({
        title: "Error",
        description: (error as Error).message || "Failed to fetch rounds",
        variant: "destructive",
      });
    } finally {
      setIsLoadingRounds(false);
    }
  };

  const fetchTopics = async () => {
    setIsLoadingTopics(true);
    try {
      const data = await apiService.getAdminCourses(); // Changed to getAdminCourses
      setTopics(data);
    } catch (error) {
      console.error("Error fetching topics:", error);
      toast({
        title: "Error",
        description: (error as Error).message || "Failed to fetch courses", // Changed description
        variant: "destructive",
      });
    } finally {
      setIsLoadingTopics(false);
    }
  };

  const handleCreateTopic = async () => {
    if (!topicFormData.name || !topicFormData.name.trim()) {
      toast({
        title: "Validation Error",
        description: "Course name is required",
        variant: "destructive",
      });
      return;
    }

    try {
      if (editingTopic) {
        await apiService.updateAdminCourse(editingTopic.id, topicFormData); // Changed to updateAdminCourse
        toast({
          title: "Success",
          description: "Course updated successfully", // Changed description
        });
      } else {
        await apiService.createAdminCourse(topicFormData); // Changed to createAdminCourse
        toast({
          title: "Success",
          description: "Course created successfully", // Changed description
        });
      }
      setShowTopicForm(false);
      setEditingTopic(null);
      setTopicFormData({ name: "", description: "" });
      fetchTopics();
    } catch (error: any) {
      toast({
        title: "Error",
        description: error.message || "Failed to save topic",
        variant: "destructive",
      });
    }
  };

  const handleDeleteTopic = async (id: number) => {
    if (!confirm("Are you sure you want to delete this course? This will also delete all levels, rounds and questions associated with it.")) return; // Changed confirmation message

    try {
      await apiService.deleteAdminCourse(id);
      toast({
        title: "Success",
        description: "Course deleted successfully", // Changed description
      });
      fetchTopics();
      // If we deleted the expanded course, reset view
      if (expandedCourseId === id) {
        setExpandedCourseId(null);
        setSelectedLevel(null);
        setSelectedRound(null);
      }
    } catch (error: any) {
      toast({
        title: "Error",
        description: error.message || "Failed to delete course", // Changed description
        variant: "destructive",
      });
    }
  };

  const handleCreateRound = async () => {
    if (!roundFormData.name || !roundFormData.name.trim()) {
      toast({
        title: "Validation Error",
        description: "Round name is required",
        variant: "destructive",
      });
      return;
    }

    if (!expandedCourseId || !selectedLevel) return;

    try {
      if (editingRound) {
        await apiService.updateAdminRound(editingRound.id, {
          ...roundFormData,
          topic: expandedCourseId,
        });
        toast({
          title: "Success",
          description: "Round updated successfully",
        });
      } else {
        await apiService.createAdminRound({
          ...roundFormData,
          topic: expandedCourseId,
          level: selectedLevel,
        });
        toast({
          title: "Success",
          description: "Round created successfully",
        });
      }
      setShowRoundForm(false);
      setEditingRound(null);
      setRoundFormData({ name: "", level: selectedLevel as any });
      fetchRounds(expandedCourseId, selectedLevel);
    } catch (error: any) {
      toast({
        title: "Error",
        description: error.message || "Failed to save round",
        variant: "destructive",
      });
    }
  };

  const handleDeleteRound = async (id: number) => {
    if (!confirm("Are you sure you want to delete this round? All associated questions will also be deleted.")) return;

    try {
      await apiService.deleteAdminRound(id);
      toast({
        title: "Success",
        description: "Round deleted successfully",
      });
      if (expandedCourseId && selectedLevel) {
        fetchRounds(expandedCourseId, selectedLevel);
      }
    } catch (error: any) {
      toast({
        title: "Error",
        description: error.message || "Failed to delete round",
        variant: "destructive",
      });
    }
  };

  const handleCreateQuestion = async () => {
    // Validate based on source_type
    if (!expandedCourseId) {
      toast({
        title: "Validation Error",
        description: "Please select a course",
        variant: "destructive",
      });
      return;
    }

    if (!selectedRound) {
      toast({
        title: "Validation Error",
        description: "Please select a round",
        variant: "destructive",
      });
      return;
    }

    if (questionFormData.source_type === "MANUAL") {
      if (!questionFormData.question_text || !questionFormData.ideal_answer) {
        toast({
          title: "Validation Error",
          description: "Question text and ideal answer are required for manual definition",
          variant: "destructive",
        });
        return;
      }
    } else if (questionFormData.source_type === "LINK") {
      if (!questionFormData.reference_links || !questionFormData.reference_links.trim()) {
        toast({
          title: "Validation Error",
          description: "Reference links are required when using link-based definition",
          variant: "destructive",
        });
      }
    }

    try {
      if (editingQuestion) {
        await apiService.updateAdminQuestion(editingQuestion.id, {
          source_type: questionFormData.source_type,
          question_text: questionFormData.source_type === "MANUAL" ? questionFormData.question_text : undefined,
          ideal_answer: questionFormData.source_type === "MANUAL" ? questionFormData.ideal_answer : undefined,
          difficulty: questionFormData.difficulty,
          is_active: questionFormData.is_active,
          reference_links: questionFormData.source_type === "LINK" ? questionFormData.reference_links : (questionFormData.reference_links || undefined),
          round: selectedRound.id,
        });
        toast({
          title: "Success",
          description: "Question updated successfully",
        });
      } else {
        await apiService.createAdminQuestion({
          topic: expandedCourseId,
          round: selectedRound.id,
          source_type: questionFormData.source_type,
          question_text: questionFormData.source_type === "MANUAL" ? questionFormData.question_text : undefined,
          ideal_answer: questionFormData.source_type === "MANUAL" ? questionFormData.ideal_answer : undefined,
          difficulty: questionFormData.difficulty,
          is_active: questionFormData.is_active,
          reference_links: questionFormData.source_type === "LINK" ? questionFormData.reference_links : (questionFormData.reference_links || undefined),
        });
        toast({
          title: "Success",
          description: "Question created successfully",
        });
      }

      setShowQuestionForm(false);
      setEditingQuestion(null);
      setQuestionFormData({
        source_type: "MANUAL",
        topic: "",
        round: "",
        question_text: "",
        ideal_answer: "",
        difficulty: "MEDIUM",
        reference_links: "",
        is_active: true,
      });
      fetchQuestions();
    } catch (error: any) {
      toast({
        title: "Error",
        description: error.message || "Failed to save question",
        variant: "destructive",
      });
    }
  };

  const handleDeleteQuestion = async (id: number) => {
    if (!confirm("Are you sure you want to delete this question?")) return;

    try {
      await apiService.deleteAdminQuestion(id);
      toast({
        title: "Success",
        description: "Question deleted successfully",
      });
      fetchQuestions();
    } catch (error: any) {
      toast({
        title: "Error",
        description: error.message || "Failed to delete question",
        variant: "destructive",
      });
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground p-4 md:p-6 font-sans">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8 flex justify-between items-start">
          <div>
            <h1 className="text-4xl font-bold text-ohg-navy mb-2 tracking-tight">
              Admin Dashboard
            </h1>
            <p className="text-muted-foreground">
              Manage interviews, questions, and user results
            </p>
          </div>
          <Button
            variant="outline"
            onClick={handleLogout}
            className="border-ohg-navy text-ohg-navy hover:bg-ohg-navy hover:text-white"
          >
            Logout
          </Button>
        </div>

        <Tabs defaultValue="users" className="space-y-6" onValueChange={setActiveTab}>
          <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 mb-8">
            <TabsList className="bg-transparent p-0 gap-2 border-none flex-wrap">
              <TabsTrigger
                value="users"
                className="
                  data-[state=active]:bg-white data-[state=active]:text-ohg-navy data-[state=active]:shadow-sm data-[state=active]:border-gray-200
                  data-[state=inactive]:bg-transparent data-[state=inactive]:text-gray-500 data-[state=inactive]:hover:bg-blue-50 data-[state=inactive]:hover:text-ohg-navy
                  border border-transparent
                  px-6 py-2 rounded-lg text-sm font-semibold transition-all duration-200 flex items-center gap-2
                "
              >
                <Users className="h-4 w-4" />
                Users & Results
              </TabsTrigger>
              <TabsTrigger
                value="topics"
                className="
                  data-[state=active]:bg-white data-[state=active]:text-ohg-navy data-[state=active]:shadow-sm data-[state=active]:border-gray-200
                  data-[state=inactive]:bg-transparent data-[state=inactive]:text-gray-500 data-[state=inactive]:hover:bg-blue-50 data-[state=inactive]:hover:text-ohg-navy
                  border border-transparent
                  px-6 py-2 rounded-lg text-sm font-semibold transition-all duration-200 flex items-center gap-2
                "
              >
                <BookOpen className="h-4 w-4" />
                Courses
              </TabsTrigger>
            </TabsList>

            {activeTab === 'topics' && expandedCourseId === null && (
              <div className="flex bg-white/50 border border-gray-200/50 p-1 rounded-lg self-end md:self-auto">
                <Button
                  variant="ghost"
                  size="sm"
                  className={`h-9 w-9 p-0 rounded-md transition-all duration-200 ${viewMode === "list"
                    ? "bg-white text-ohg-navy shadow-sm ring-1 ring-gray-200"
                    : "text-gray-400 hover:text-ohg-navy hover:bg-blue-50"
                    }`}
                  onClick={() => setViewMode("list")}
                >
                  <LayoutList className="h-4 w-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className={`h-9 w-9 p-0 rounded-md transition-all duration-200 ${viewMode === "grid"
                    ? "bg-white text-ohg-navy shadow-sm ring-1 ring-gray-200"
                    : "text-gray-400 hover:text-ohg-navy hover:bg-blue-50"
                    }`}
                  onClick={() => setViewMode("grid")}
                >
                  <LayoutGrid className="h-4 w-4" />
                </Button>
              </div>
            )}
          </div>

          <TabsContent value="users">
            <div className="space-y-4 animate-in fade-in slide-in-from-bottom-4 duration-500">
              <div className="flex justify-end">
                <Button
                  onClick={() => setShowUserForm(!showUserForm)}
                  className="bg-ohg-navy text-white hover:bg-ohg-navy/90"
                >
                  <Plus className="h-4 w-4 mr-2" />
                  Add User
                </Button>
              </div>

              {showUserForm && (
                <Card className="p-4 md:p-6 bg-white/80 backdrop-blur-lg border-border shadow-soft animate-slide-up">
                  <h3 className="text-lg font-semibold text-foreground mb-4">
                    Create New User
                  </h3>
                  <div className="space-y-4">
                    <div>
                      <label className="text-sm text-foreground mb-2 block">
                        Username <span className="text-destructive">*</span>
                      </label>
                      <Input
                        placeholder="Enter username"
                        value={userFormData.username}
                        onChange={(e) => setUserFormData({ ...userFormData, username: e.target.value })}
                        className="bg-input border-border"
                      />
                    </div>
                    <div>
                      <label className="text-sm text-foreground mb-2 block">
                        Password <span className="text-destructive">*</span>
                      </label>
                      <div className="relative">
                        <Input
                          type={showPassword ? "text" : "password"}
                          placeholder="Enter password"
                          value={userFormData.password}
                          onChange={(e) => setUserFormData({ ...userFormData, password: e.target.value })}
                          className="bg-input border-border pr-10"
                        />
                        <button
                          type="button"
                          onClick={() => setShowPassword(!showPassword)}
                          className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors focus:outline-none"
                        >
                          {showPassword ? (
                            <EyeOff className="h-4 w-4" />
                          ) : (
                            <Eye className="h-4 w-4" />
                          )}
                        </button>
                      </div>
                    </div>
                    <div>
                      <label className="text-sm text-foreground mb-2 block">
                        Email <span className="text-destructive">*</span>
                      </label>
                      <Input
                        type="email"
                        placeholder="Enter email"
                        value={userFormData.email}
                        onChange={(e) => setUserFormData({ ...userFormData, email: e.target.value })}
                        className="bg-input border-border"
                      />
                    </div>
                    <div>
                      <label className="text-sm text-foreground mb-2 block">
                        Name (Optional)
                      </label>
                      <Input
                        placeholder="Enter full name"
                        value={userFormData.name}
                        onChange={(e) => setUserFormData({ ...userFormData, name: e.target.value })}
                        className="bg-input border-border"
                      />
                    </div>
                    <div>
                      <label className="text-sm text-foreground mb-2 block">
                        Role <span className="text-destructive">*</span>
                      </label>
                      <Select
                        value={userFormData.role}
                        onValueChange={(value: 'ADMIN' | 'USER') =>
                          setUserFormData({ ...userFormData, role: value })
                        }
                      >
                        <SelectTrigger className="bg-input border-border text-foreground">
                          <SelectValue placeholder="Select role" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="USER">Student (User)</SelectItem>
                          <SelectItem value="ADMIN">Administrator</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    {userFormData.role === 'USER' && (
                      <div>
                        <label className="text-sm text-foreground mb-2 block">
                          Access Type <span className="text-destructive">*</span>
                        </label>
                        <Select
                          value={userFormData.access_type}
                          onValueChange={(value: 'TRIAL' | 'FULL') =>
                            setUserFormData({ ...userFormData, access_type: value })
                          }
                        >
                          <SelectTrigger className="bg-input border-border text-foreground">
                            <SelectValue placeholder="Select access type" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="TRIAL">Trial - One Free Interview</SelectItem>
                            <SelectItem value="FULL">Full Access</SelectItem>
                          </SelectContent>
                        </Select>
                        <p className="text-xs text-muted-foreground mt-1">
                          {userFormData.access_type === 'TRIAL'
                            ? 'User will get one free interview'
                            : 'User can create unlimited interview sessions'}
                        </p>
                      </div>
                    )}
                    <div className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        id="is_active"
                        checked={userFormData.is_active}
                        onChange={(e) => setUserFormData({ ...userFormData, is_active: e.target.checked })}
                        className="w-4 h-4 rounded border-border bg-input text-primary focus:ring-primary"
                      />
                      <label htmlFor="is_active" className="text-sm text-foreground cursor-pointer">
                        Account is active
                      </label>
                    </div>
                    <div className="flex gap-2">
                      <Button
                        onClick={() => {
                          setShowUserForm(false);
                          setUserFormData({
                            username: "",
                            password: "",
                            email: "",
                            name: "",
                            is_active: true,
                            role: 'USER',
                            access_type: 'TRIAL',
                          });
                        }}
                        variant="outline"
                        className="flex-1"
                        disabled={isCreatingUser}
                      >
                        Cancel
                      </Button>
                      <Button
                        onClick={handleCreateUser}
                        className="flex-1 bg-ohg-navy hover:bg-ohg-navy/90 text-white"
                        disabled={isCreatingUser}
                      >
                        {isCreatingUser ? "Creating..." : "Create User"}
                      </Button>
                    </div>
                  </div>
                </Card>
              )}

              <Card className="p-4 md:p-6 bg-white/60 backdrop-blur-lg border-border shadow-sm">
                {isLoadingUsers ? (
                  <div className="text-center py-8 text-muted-foreground">
                    Loading users...
                  </div>
                ) : users.length === 0 ? (
                  <div className="text-center py-8 text-muted-foreground">
                    No users found. Create your first user above.
                  </div>
                ) : (
                  <div className="overflow-x-auto">
                    <Table>
                      <TableHeader>
                        <TableRow className="border-border hover:bg-muted/20">
                          <TableHead className="text-foreground">Username</TableHead>
                          <TableHead className="text-foreground">Name</TableHead>
                          <TableHead className="text-foreground">Email</TableHead>
                          <TableHead className="text-foreground">Role</TableHead>
                          {users.some(u => u.role === 'USER') && (
                            <>
                              <TableHead className="text-foreground">Access Type</TableHead>
                              <TableHead className="text-foreground">Trial Used</TableHead>
                            </>
                          )}
                          <TableHead className="text-foreground">Created</TableHead>
                          <TableHead className="text-foreground">Actions</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {Array.isArray(users) && users.map((user) => (
                          <TableRow key={user.id} className="border-border hover:bg-muted/20">
                            <TableCell className="text-foreground font-medium">{user.username}</TableCell>
                            <TableCell className="text-foreground">{user.name || "-"}</TableCell>
                            <TableCell className="text-muted-foreground">{user.email}</TableCell>
                            <TableCell>
                              <span className={`px-2 py-1 rounded text-xs ${user.role === 'ADMIN' ? 'bg-purple-500/20 text-purple-500' : 'bg-blue-500/20 text-blue-500'}`}>
                                {user.role === 'ADMIN' ? 'Admin' : 'User'}
                              </span>
                            </TableCell>

                            {users.some(u => u.role === 'USER') && (
                              <>
                                <TableCell>
                                  {user.role === 'ADMIN' ? (
                                    <span className="text-muted-foreground text-xs">-</span>
                                  ) : (
                                    <span
                                      className={`px-2 py-1 rounded text-xs ${user.access_type === 'FULL'
                                        ? "bg-primary/20 text-primary font-medium"
                                        : "bg-secondary/20 text-secondary"
                                        }`}
                                    >
                                      {user.access_type === 'FULL' ? 'Full Access' : 'Trial'}
                                    </span>
                                  )}
                                </TableCell>
                                <TableCell>
                                  {user.role === 'ADMIN' ? (
                                    <span className="text-muted-foreground text-xs">-</span>
                                  ) : (
                                    <span
                                      className={`px-2 py-1 rounded text-xs ${user.has_used_trial
                                        ? "bg-secondary/20 text-secondary"
                                        : "bg-primary/20 text-primary"
                                        }`}
                                    >
                                      {user.has_used_trial ? "Yes" : "No"}
                                    </span>
                                  )}
                                </TableCell>
                              </>
                            )}

                            <TableCell>
                              <span
                                className={`px-2 py-1 rounded text-xs ${user.is_active
                                  ? "bg-primary/20 text-primary"
                                  : "bg-muted text-muted-foreground"
                                  }`}
                              >
                                {user.is_active ? "Active" : "Inactive"}
                              </span>
                            </TableCell>
                            <TableCell className="text-muted-foreground text-sm">
                              {new Date(user.created_at).toLocaleDateString()}
                            </TableCell>
                            <TableCell>
                              <div className="flex gap-2">
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  className="hover:bg-primary/10"
                                  onClick={() => handleViewUserDetails(user)}
                                >
                                  View Details
                                </Button>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  className="hover:bg-destructive/10"
                                  onClick={() => handleDeleteUser(user.id)}
                                >
                                  <Trash2 className="h-4 w-4 text-destructive" />
                                </Button>
                              </div>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                )}
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="topics">
            <div className="space-y-4">
              {expandedCourseId === null ? (
                /* -------------------------------------------------------------------------- */
                /*                            COURSE LIST VIEW                                */
                /* -------------------------------------------------------------------------- */
                <>
                  <div className="flex justify-end mb-6">
                    <Button
                      onClick={() => setShowTopicForm(!showTopicForm)}
                      className="bg-ohg-navy hover:bg-ohg-navy/90 text-white shadow-lg shadow-ohg-navy/20"
                    >
                      <Plus className="h-4 w-4 mr-2" />
                      Add Course
                    </Button>
                  </div>

                  {showTopicForm && (
                    <Card className="p-4 md:p-6 bg-white/80 backdrop-blur-lg border-border shadow-soft animate-slide-up mb-4">
                      <h3 className="text-lg font-semibold text-foreground mb-4">
                        {editingTopic ? "Edit Course" : "Add New Course"}
                      </h3>
                      <div className="space-y-4">
                        <div>
                          <label className="text-sm text-foreground mb-2 block">Course Name *</label>
                          <Input
                            placeholder="e.g., Machine Learning, React, System Design"
                            className="bg-input border-border text-foreground"
                            value={topicFormData.name}
                            onChange={(e) => setTopicFormData({ ...topicFormData, name: e.target.value })}
                          />
                        </div>
                        <div>
                          <label className="text-sm text-foreground mb-2 block">Description (Optional)</label>
                          <Textarea
                            placeholder="Brief description of this topic..."
                            className="bg-input border-border min-h-[80px]"
                            value={topicFormData.description}
                            onChange={(e) => setTopicFormData({ ...topicFormData, description: e.target.value })}
                          />
                        </div>
                        <div className="flex gap-2">
                          <Button
                            onClick={() => {
                              setShowTopicForm(false);
                              setEditingTopic(null);
                              setTopicFormData({ name: "", description: "" });
                            }}
                            variant="outline"
                            className="flex-1"
                          >
                            Cancel
                          </Button>
                          <Button
                            onClick={handleCreateTopic}
                            className="flex-1 bg-ohg-navy hover:bg-ohg-navy/90 text-white"
                          >
                            {editingTopic ? "Update Course" : "Save Course"}
                          </Button>
                        </div>
                      </div>
                    </Card>
                  )}

                  {viewMode === "list" ? (
                    <Card className="p-4 md:p-6 bg-white/60 backdrop-blur-lg border-border shadow-sm">
                      {isLoadingTopics ? (
                        <div className="text-center py-8 text-muted-foreground">Loading courses...</div>
                      ) : topics.length === 0 ? (
                        <div className="text-center py-8 text-muted-foreground">No courses found. Add your first course above.</div>
                      ) : (
                        <div className="overflow-x-auto">
                          <Table>
                            <TableHeader>
                              <TableRow className="border-border hover:bg-muted/20">
                                <TableHead className="text-foreground">Course Name</TableHead>
                                <TableHead className="text-foreground">Description</TableHead>
                                <TableHead className="text-foreground">Created</TableHead>
                                <TableHead className="text-foreground">Actions</TableHead>
                              </TableRow>
                            </TableHeader>
                            <TableBody>
                              {topics.map((topic) => (
                                <TableRow
                                  key={topic.id}
                                  className="border-border hover:bg-muted/20 cursor-pointer group"
                                  onClick={() => setExpandedCourseId(topic.id)}
                                >
                                  <TableCell className="text-foreground font-medium group-hover:text-primary transition-colors">
                                    {topic.name}
                                  </TableCell>
                                  <TableCell className="text-foreground">
                                    {topic.description || <span className="text-muted-foreground">No description</span>}
                                  </TableCell>
                                  <TableCell className="text-muted-foreground text-sm">
                                    {new Date(topic.created_at).toLocaleDateString()}
                                  </TableCell>
                                  <TableCell onClick={(e) => e.stopPropagation()}>
                                    <div className="flex gap-2">
                                      <Button
                                        variant="ghost"
                                        size="sm"
                                        className="hover:bg-primary/10"
                                        onClick={() => {
                                          setEditingTopic(topic);
                                          setTopicFormData({
                                            name: topic.name,
                                            description: topic.description || "",
                                          });
                                          setShowTopicForm(true);
                                        }}
                                      >
                                        <Edit className="h-4 w-4" />
                                      </Button>
                                      <Button
                                        variant="ghost"
                                        size="sm"
                                        className="hover:bg-destructive/10"
                                        onClick={() => handleDeleteTopic(topic.id)}
                                      >
                                        <Trash2 className="h-4 w-4 text-destructive" />
                                      </Button>
                                    </div>
                                  </TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </div>
                      )}
                    </Card>
                  ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                      {isLoadingTopics ? (
                        <div className="col-span-full text-center py-8 text-muted-foreground">Loading courses...</div>
                      ) : topics.length === 0 ? (
                        <div className="col-span-full text-center py-8 text-muted-foreground">No courses found.</div>
                      ) : (
                        topics.map((topic) => (
                          <Card
                            key={topic.id}
                            className="p-4 bg-white/60 backdrop-blur-lg border-border shadow-sm hover:shadow-md transition-shadow cursor-pointer relative group flex flex-col justify-between"
                            onClick={() => setExpandedCourseId(topic.id)}
                          >
                            <div>
                              <div className="flex justify-between items-start mb-2">
                                <h3 className="font-semibold text-foreground text-lg group-hover:text-primary transition-colors">
                                  {topic.name}
                                </h3>
                              </div>
                              <p className="text-sm text-muted-foreground mb-4 line-clamp-2">
                                {topic.description || "No description provided."}
                              </p>
                            </div>

                            <div className="flex justify-between items-end mt-auto pt-2 border-t border-border/50">
                              <span />
                              <div className="flex gap-1" onClick={(e) => e.stopPropagation()}>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  className="h-8 w-8 p-0 hover:bg-primary/10"
                                  onClick={() => {
                                    setEditingTopic(topic);
                                    setTopicFormData({
                                      name: topic.name,
                                      description: topic.description || "",
                                    });
                                    setShowTopicForm(true);
                                  }}
                                >
                                  <Edit className="h-4 w-4" />
                                </Button>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  className="h-8 w-8 p-0 hover:bg-destructive/10"
                                  onClick={() => handleDeleteTopic(topic.id)}
                                >
                                  <Trash2 className="h-4 w-4 text-destructive" />
                                </Button>
                              </div>
                            </div>
                          </Card>
                        ))
                      )}
                    </div>
                  )}
                </>
              ) : (
                /* -------------------------------------------------------------------------- */
                /*                          COURSE DETAIL / QUESTIONS VIEW                    */
                /* -------------------------------------------------------------------------- */
                <div className="space-y-4">
                  {selectedLevel === null ? (
                    /* -------------------------------------------------------------------------- */
                    /*                            LEVEL SELECTION VIEW                            */
                    /* -------------------------------------------------------------------------- */
                    <div className="space-y-4 animate-in fade-in slide-in-from-right-4 duration-300">
                      <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center gap-2">
                          <Button
                            variant="ghost"
                            size="sm"
                            className="p-1 h-8 w-8 rounded-full hover:bg-muted"
                            onClick={() => setExpandedCourseId(null)}
                          >
                            <ArrowLeft className="h-5 w-5 text-foreground" />
                          </Button>
                          <h3 className="text-xl font-semibold text-foreground">
                            {topics.find(t => t.id === expandedCourseId)?.name} <span className="text-muted-foreground mx-2">/</span> Select Difficulty
                          </h3>
                        </div>
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                        {['BEGINNER', 'INTERMEDIATE', 'ADVANCED'].map((level) => (
                          <Card
                            key={level}
                            className="group p-6 hover:shadow-lg transition-all cursor-pointer border-t-4 border-t-transparent hover:border-t-ohg-navy bg-white/60 backdrop-blur-sm"
                            onClick={() => {
                              setSelectedLevel(level as any);
                              fetchRounds(expandedCourseId!, level);
                            }}
                          >
                            <div className="mb-4 p-3 rounded-full bg-primary/10 w-fit group-hover:bg-primary/20 transition-colors">
                              {level === 'BEGINNER' && <div className="h-6 w-6 rounded-full border-2 border-green-500" />}
                              {level === 'INTERMEDIATE' && <div className="h-6 w-6 rounded-full border-2 border-amber-500" />}
                              {level === 'ADVANCED' && <div className="h-6 w-6 rounded-full border-2 border-red-500" />}
                            </div>
                            <h3 className="text-lg font-bold text-ohg-navy mb-2">{level}</h3>
                            <p className="text-sm text-muted-foreground">Manage rounds and questions for {level.toLowerCase()} difficulty level.</p>
                          </Card>
                        ))}
                      </div>
                    </div>
                  ) : selectedRound === null ? (
                    /* -------------------------------------------------------------------------- */
                    /*                            ROUND LIST VIEW                                 */
                    /* -------------------------------------------------------------------------- */
                    <div className="space-y-4 animate-in fade-in slide-in-from-right-4 duration-300">
                      <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center gap-2">
                          <Button
                            variant="ghost"
                            size="sm"
                            className="p-1 h-8 w-8 rounded-full hover:bg-muted"
                            onClick={() => setSelectedLevel(null)}
                          >
                            <ArrowLeft className="h-5 w-5 text-foreground" />
                          </Button>
                          <h3 className="text-xl font-semibold text-foreground flex items-center">
                            {topics.find(t => t.id === expandedCourseId)?.name}
                            <span className="text-muted-foreground mx-2">/</span>
                            {selectedLevel}
                            <span className="text-muted-foreground mx-2">/</span>
                            Rounds
                          </h3>
                        </div>
                        <Button onClick={() => {
                          setRoundFormData({ name: "Round _", level: selectedLevel as any });
                          setShowRoundForm(true);
                        }} className="bg-ohg-navy text-white shadow-lg">
                          <Plus className="h-4 w-4 mr-2" />
                          Add Round
                        </Button>
                      </div>

                      {showRoundForm && (
                        <Card className="p-4 md:p-6 bg-white/80 backdrop-blur-lg border-border shadow-soft animate-slide-up mb-4">
                          <h3 className="text-lg font-semibold text-foreground mb-4">
                            {editingRound ? "Edit Round" : "Add New Round"}
                          </h3>
                          <div className="space-y-4">
                            <div>
                              <label className="text-sm text-foreground mb-2 block">Round Name *</label>
                              <Input
                                placeholder="e.g., Round 1, Technical Screening, System Design"
                                className="bg-input border-border text-foreground"
                                value={roundFormData.name}
                                onChange={(e) => setRoundFormData({ ...roundFormData, name: e.target.value })}
                              />
                            </div>
                            <div className="flex gap-2">
                              <Button
                                onClick={() => {
                                  setShowRoundForm(false);
                                  setEditingRound(null);
                                  setRoundFormData({ name: "", level: selectedLevel as any });
                                }}
                                variant="outline"
                                className="flex-1"
                              >
                                Cancel
                              </Button>
                              <Button
                                onClick={handleCreateRound}
                                className="flex-1 bg-ohg-navy hover:bg-ohg-navy/90 text-white"
                              >
                                {editingRound ? "Update Round" : "Save Round"}
                              </Button>
                            </div>
                          </div>
                        </Card>
                      )}

                      <div className="grid gap-4">
                        {isLoadingRounds ? (
                          <div className="text-center py-8 text-muted-foreground">Loading rounds...</div>
                        ) : rounds.length === 0 ? (
                          <div className="text-center py-12 border-2 border-dashed border-muted rounded-xl">
                            <h4 className="text-lg font-medium text-foreground mb-2">No rounds found</h4>
                            <p className="text-muted-foreground mb-4">Create a round to start adding questions.</p>
                            <Button variant="outline" onClick={() => {
                              setRoundFormData({ name: "Round _", level: selectedLevel as any });
                              setShowRoundForm(true);
                            }}>Add First Round</Button>
                          </div>
                        ) : (
                          rounds.map(round => (
                            <Card
                              key={round.id}
                              className="p-4 flex justify-between items-center hover:shadow-md transition-shadow cursor-pointer border-l-4 border-l-ohg-navy"
                              onClick={() => {
                                setSelectedRound(round);
                                setIsLoadingQuestions(true);
                                apiService.getAdminQuestions(expandedCourseId!, round.id)
                                  .then((data) => setQuestions(data))
                                  .finally(() => setIsLoadingQuestions(false));
                              }}
                            >
                              <div>
                                <h4 className="font-semibold text-lg">{round.name}</h4>
                                <p className="text-sm text-muted-foreground">{round.question_count} questions</p>
                              </div>
                              <div className="flex gap-2" onClick={e => e.stopPropagation()}>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  onClick={() => {
                                    setEditingRound(round);
                                    setRoundFormData({ name: round.name, level: round.level });
                                    setShowRoundForm(true);
                                  }}
                                >
                                  <Edit className="h-4 w-4" />
                                </Button>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  className="text-destructive hover:bg-destructive/10"
                                  onClick={() => handleDeleteRound(round.id)}
                                >
                                  <Trash2 className="h-4 w-4" />
                                </Button>
                              </div>
                            </Card>
                          ))
                        )}
                      </div>
                    </div>
                  ) : (
                    /* -------------------------------------------------------------------------- */
                    /*                            QUESTION LIST VIEW                              */
                    /* -------------------------------------------------------------------------- */
                    <div className="space-y-4 animate-in fade-in slide-in-from-right-4 duration-300">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <Button
                            variant="ghost"
                            size="sm"
                            className="p-1 h-8 w-8 rounded-full hover:bg-muted"
                            onClick={() => setSelectedRound(null)}
                          >
                            <ArrowLeft className="h-5 w-5 text-foreground" />
                          </Button>
                          <h3 className="text-xl font-semibold text-foreground flex items-center">
                            {selectedLevel} <span className="text-muted-foreground mx-2">/</span>
                            {selectedRound.name} <span className="text-muted-foreground mx-2">/</span>
                            Questions
                          </h3>
                        </div>
                        <Button
                          onClick={() => {
                            setQuestionFormData({
                              ...questionFormData,
                              topic: expandedCourseId!.toString(),
                              round: selectedRound.id.toString(), // Pre-fill round
                            });
                            setShowQuestionForm(true);
                          }}
                          className="bg-ohg-navy hover:bg-ohg-navy/90 text-white shadow-lg shadow-ohg-navy/20"
                        >
                          <Plus className="h-4 w-4 mr-2" />
                          Add Question
                        </Button>
                      </div>

                      {showQuestionForm && (
                        <Card className="p-4 md:p-6 bg-white/80 backdrop-blur-lg border-border shadow-soft animate-slide-up mb-4">
                          <h3 className="text-lg font-semibold text-foreground mb-4">
                            {editingQuestion ? "Edit Question" : "Add New Question"}
                          </h3>
                          <div className="space-y-4">
                            <div>
                              <label className="text-sm text-foreground mb-2 block">Source Type</label>
                              <div className="flex gap-4 mb-4">
                                <div className="flex items-center space-x-2">
                                  <input
                                    type="radio"
                                    id="manual"
                                    value="MANUAL"
                                    checked={questionFormData.source_type === "MANUAL"}
                                    onChange={() => setQuestionFormData({ ...questionFormData, source_type: "MANUAL", round: selectedRound.id.toString() })}
                                    className="text-ohg-navy focus:ring-ohg-navy"
                                  />
                                  <label htmlFor="manual" className="text-sm cursor-pointer">Manual Entry</label>
                                </div>
                                <div className="flex items-center space-x-2">
                                  <input
                                    type="radio"
                                    id="link"
                                    value="LINK"
                                    checked={questionFormData.source_type === "LINK"}
                                    onChange={() => setQuestionFormData({ ...questionFormData, source_type: "LINK", round: selectedRound.id.toString() })}
                                    className="text-ohg-navy focus:ring-ohg-navy"
                                  />
                                  <label htmlFor="link" className="text-sm cursor-pointer">Link / File</label>
                                </div>
                              </div>
                            </div>
                            {/* Hidden Topic/Round Select - Contextually set */}
                            <div className="hidden">
                              <Select
                                value={questionFormData.topic}
                                onValueChange={(value) => setQuestionFormData({ ...questionFormData, topic: value, round: selectedRound.id.toString() })}
                                disabled={true}
                              >
                              </Select>
                            </div>

                            {questionFormData.source_type === "MANUAL" ? (
                              <>
                                <div>
                                  <label className="text-sm text-foreground mb-2 block">Question Text *</label>
                                  <Textarea
                                    placeholder="Enter the interview question"
                                    className="bg-input border-border min-h-[80px]"
                                    value={questionFormData.question_text}
                                    onChange={(e) => setQuestionFormData({ ...questionFormData, question_text: e.target.value, round: selectedRound.id.toString() })}
                                  />
                                </div>
                                <div>
                                  <label className="text-sm text-foreground mb-2 block">Ideal Answer / Key Points *</label>
                                  <Textarea
                                    placeholder="Enter the ideal answer or key points to look for"
                                    className="bg-input border-border min-h-[120px]"
                                    value={questionFormData.ideal_answer}
                                    onChange={(e) => setQuestionFormData({ ...questionFormData, ideal_answer: e.target.value, round: selectedRound.id.toString() })}
                                  />
                                </div>
                              </>
                            ) : (
                              <div>
                                <label className="text-sm text-foreground mb-2 block">Reference Link *</label>
                                <Input
                                  placeholder="https://..."
                                  className="bg-input border-border"
                                  value={questionFormData.reference_links}
                                  onChange={(e) => setQuestionFormData({ ...questionFormData, reference_links: e.target.value, round: selectedRound.id.toString() })}
                                />
                                <p className="text-xs text-muted-foreground mt-1">Provide a link to the question content.</p>
                              </div>
                            )}

                            {/* Hidden Difficulty and Active Checkbox - Defaults used */}
                            <div className="hidden">
                              <Select
                                value={questionFormData.difficulty}
                                onValueChange={(value: "EASY" | "MEDIUM" | "HARD") =>
                                  setQuestionFormData({ ...questionFormData, difficulty: value, round: selectedRound.id.toString() })
                                }
                              >
                                <SelectTrigger className="bg-input border-border text-foreground">
                                  <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                  <SelectItem value="EASY">Easy</SelectItem>
                                  <SelectItem value="MEDIUM">Medium</SelectItem>
                                  <SelectItem value="HARD">Hard</SelectItem>
                                </SelectContent>
                              </Select>
                              <input
                                type="checkbox"
                                id="qs_active"
                                checked={questionFormData.is_active}
                                onChange={(e) => setQuestionFormData({ ...questionFormData, is_active: e.target.checked, round: selectedRound.id.toString() })}
                              />
                            </div>

                            <div className="flex gap-2 pt-2">
                              <Button
                                onClick={() => {
                                  setShowQuestionForm(false);
                                  setEditingQuestion(null);
                                }}
                                variant="outline"
                                className="flex-1"
                              >
                                Cancel
                              </Button>
                              <Button
                                onClick={handleCreateQuestion}
                                className="flex-1 bg-ohg-navy hover:bg-ohg-navy/90 text-white"
                              >
                                {editingQuestion ? "Update Question" : "Save Question"}
                              </Button>
                            </div>
                          </div>
                        </Card>
                      )}

                      <Card className="p-4 md:p-6 bg-white/60 backdrop-blur-lg border-border shadow-sm">
                        {questions.length === 0 ? (
                          <div className="text-center py-8 text-muted-foreground">
                            No questions for this round yet.
                          </div>
                        ) : (
                          <div className="overflow-x-auto">
                            <Table>
                              <TableHeader>
                                <TableRow className="border-border hover:bg-muted/20">
                                  <TableHead className="w-[40%] text-foreground">Question</TableHead>
                                  <TableHead className="text-foreground">Difficulty</TableHead>
                                  <TableHead className="text-foreground">Type</TableHead>
                                  <TableHead className="text-foreground">Status</TableHead>
                                  <TableHead className="text-right text-foreground">Actions</TableHead>
                                </TableRow>
                              </TableHeader>
                              <TableBody>
                                {questions
                                  .map((q) => (
                                    <TableRow key={q.id} className="border-border hover:bg-muted/20">
                                      <TableCell className="font-medium max-w-md truncate text-foreground" title={q.question_text}>
                                        {q.question_text || "Link based content"}
                                      </TableCell>
                                      <TableCell>
                                        <span className={`px-2 py-0.5 rounded text-[10px] uppercase font-bold border ${q.difficulty === 'EASY' ? 'bg-green-100 text-green-700 border-green-200' :
                                          q.difficulty === 'MEDIUM' ? 'bg-amber-100 text-amber-700 border-amber-200' :
                                            'bg-red-100 text-red-700 border-red-200'
                                          }`}>
                                          {q.difficulty}
                                        </span>
                                      </TableCell>
                                      <TableCell className="text-xs text-muted-foreground">{q.source_type}</TableCell>
                                      <TableCell>
                                        <span className={`w-2 h-2 rounded-full inline-block mr-2 ${q.is_active ? 'bg-emerald-500' : 'bg-gray-300'}`} />
                                        <span className="text-xs text-muted-foreground">{q.is_active ? 'Active' : 'Inactive'}</span>
                                      </TableCell>
                                      <TableCell className="text-right">
                                        <div className="flex justify-end gap-2">
                                          <Button
                                            variant="ghost"
                                            size="sm"
                                            className="hover:bg-primary/10"
                                            onClick={() => {
                                              setEditingQuestion(q);
                                              setQuestionFormData({
                                                source_type: q.source_type || "MANUAL",
                                                topic: q.topic.toString(),
                                                round: q.round?.toString() || selectedRound.id.toString(),
                                                question_text: q.question_text,
                                                ideal_answer: q.ideal_answer,
                                                difficulty: q.difficulty,
                                                reference_links: q.reference_links || "",
                                                is_active: q.is_active,
                                              });
                                              setShowQuestionForm(true);
                                            }}
                                          >
                                            <Edit className="h-4 w-4" />
                                          </Button>
                                          <Button
                                            variant="ghost"
                                            size="sm"
                                            className="hover:bg-destructive/10"
                                            onClick={() => handleDeleteQuestion(q.id)}
                                          >
                                            <Trash2 className="h-4 w-4 text-destructive" />
                                          </Button>
                                        </div>
                                      </TableCell>
                                    </TableRow>
                                  ))}
                              </TableBody>
                            </Table>
                          </div>
                        )}
                      </Card>
                    </div>
                  )}
                </div>
              )}
            </div>
          </TabsContent>
        </Tabs>
      </div>

      {/* User Details Dialog */}
      < Dialog open={showUserDetails} onOpenChange={setShowUserDetails} >
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto bg-white backdrop-blur-xl border-border shadow-2xl">
          <DialogHeader>
            <DialogTitle className="text-2xl font-bold text-ohg-navy">
              User Details
            </DialogTitle>
            <DialogDescription className="text-muted-foreground">
              View and manage user information and interview sessions
            </DialogDescription>
          </DialogHeader>

          {selectedUser && (
            <div className="space-y-6 mt-4">
              {/* User Information */}
              <Card className="p-4 bg-card/50 border-border">
                <h3 className="text-lg font-semibold text-foreground mb-4">User Information</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="text-sm text-muted-foreground">Username</label>
                    <p className="text-foreground font-medium">{selectedUser.username}</p>
                  </div>
                  <div>
                    <label className="text-sm text-muted-foreground">Email</label>
                    <p className="text-foreground">{selectedUser.email}</p>
                  </div>
                  <div>
                    <label className="text-sm text-muted-foreground">Name</label>
                    <p className="text-foreground">{selectedUser.name || "-"}</p>
                  </div>
                  <div>
                    <label className="text-sm text-muted-foreground">Account Status</label>
                    <p className="text-foreground">
                      <span
                        className={`px-2 py-1 rounded text-xs ${selectedUser.is_active
                          ? "bg-primary/20 text-primary"
                          : "bg-muted text-muted-foreground"
                          }`}
                      >
                        {selectedUser.is_active ? "Active" : "Inactive"}
                      </span>
                    </p>
                  </div>
                  <div>
                    <label className="text-sm text-muted-foreground">Role</label>
                    <p className="text-foreground">
                      <span className={`px-2 py-1 rounded text-xs ${selectedUser.role === 'ADMIN' ? 'bg-purple-500/20 text-purple-500' : 'bg-blue-500/20 text-blue-500'}`}>
                        {selectedUser.role === 'ADMIN' ? 'Administrator' : 'Student (User)'}
                      </span>
                    </p>
                  </div>
                  {selectedUser.plain_password && (
                    <div>
                      <label className="text-sm text-muted-foreground">Current Password</label>
                      <p className="text-red-400 font-mono text-sm bg-red-950/30 p-1 rounded border border-red-900/50 inline-block">
                        {selectedUser.plain_password}
                      </p>
                    </div>
                  )}
                  {selectedUser.role === 'USER' && (
                    <div>
                      <label className="text-sm text-muted-foreground">Access Type</label>
                      <p className="text-foreground">
                        <span
                          className={`px-2 py-1 rounded text-xs ${selectedUser.access_type === 'FULL'
                            ? "bg-primary/20 text-primary font-medium"
                            : "bg-secondary/20 text-secondary"
                            }`}
                        >
                          {selectedUser.access_type === 'FULL' ? 'Full Access' : 'Trial'}
                        </span>
                      </p>
                    </div>
                  )}
                  {selectedUser.access_type === 'TRIAL' && (
                    <div>
                      <label className="text-sm text-muted-foreground">Trial Status</label>
                      <p className="text-foreground">
                        <span
                          className={`px-2 py-1 rounded text-xs ${selectedUser.has_used_trial
                            ? "bg-secondary/20 text-secondary"
                            : "bg-primary/20 text-primary"
                            }`}
                        >
                          {selectedUser.has_used_trial ? "Used" : "Available"}
                        </span>
                      </p>
                    </div>
                  )}
                  <div>
                    <label className="text-sm text-muted-foreground">Created</label>
                    <p className="text-foreground text-sm">
                      {new Date(selectedUser.created_at).toLocaleString()}
                    </p>
                  </div>
                </div>

                <div className="flex flex-col gap-4 mt-6">
                  {showPasswordChange ? (
                    <div className="p-4 rounded-lg bg-muted/20 border border-border">
                      <h4 className="text-sm font-medium text-foreground mb-3">Change Password</h4>
                      <div className="flex gap-2">
                        <Input
                          type="text"
                          placeholder="Enter new password"
                          value={newPassword}
                          onChange={(e) => setNewPassword(e.target.value)}
                          className="bg-input border-border"
                        />
                        <Button
                          onClick={handleChangePassword}
                          disabled={!newPassword || newPassword.length < 6}
                          className="whitespace-nowrap"
                        >
                          Save
                        </Button>
                        <Button
                          variant="ghost"
                          onClick={() => {
                            setShowPasswordChange(false);
                            setNewPassword("");
                          }}
                        >
                          Cancel
                        </Button>
                      </div>
                      <p className="text-xs text-muted-foreground mt-2">
                        Password must be at least 6 characters.
                      </p>
                    </div>
                  ) : null}

                  <div className="flex flex-wrap gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleToggleUserStatus(selectedUser)}
                    >
                      {selectedUser.is_active ? (
                        <>
                          <EyeOff className="h-4 w-4 mr-2" />
                          Deactivate User
                        </>
                      ) : (
                        <>
                          <Eye className="h-4 w-4 mr-2" />
                          Activate User
                        </>
                      )}
                    </Button>
                    {selectedUser.role === 'USER' && (
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={handleUpdateAccessType}
                      >
                        {selectedUser.access_type === 'FULL' ? 'Set to Trial' : 'Set to Full Access'}
                      </Button>
                    )}
                    {!showPasswordChange && (
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setShowPasswordChange(true)}
                      >
                        <Lock className="h-4 w-4 mr-2" />
                        Change Password
                      </Button>
                    )}
                    {selectedUser.has_used_trial && (
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleResetTrial(selectedUser.id)}
                      >
                        <RefreshCw className="h-4 w-4 mr-2" />
                        Reset Trial
                      </Button>
                    )}
                    <Button
                      variant="destructive"
                      size="sm"
                      onClick={() => handleDeleteUser(selectedUser.id)}
                      className="ml-auto"
                    >
                      <Trash2 className="h-4 w-4 mr-2" />
                      Delete User
                    </Button>
                  </div>
                </div>
              </Card>

              {/* Interview Sessions */}
              {/* Interview Sessions - Only for Students */}
              {selectedUser.role === 'USER' && (
                <Card className="p-4 bg-card/50 border-border">
                  <h3 className="text-lg font-semibold text-foreground mb-4">
                    Interview Sessions ({userSessions.length})
                  </h3>

                  {isLoadingSessions ? (
                    <div className="text-center py-8 text-muted-foreground">
                      Loading sessions...
                    </div>
                  ) : userSessions.length === 0 ? (
                    <div className="text-center py-8 text-muted-foreground">
                      No interview sessions found for this user.
                    </div>
                  ) : (
                    <div className="space-y-3">
                      {userSessions.map((session) => (
                        <div
                          key={session.id}
                          className="bg-muted/20 rounded-lg border border-border overflow-hidden"
                        >
                          <div className="p-4 flex flex-col gap-3">
                            <div className="flex justify-between items-start">
                              <div>
                                <div className="flex items-center gap-2 mb-1">
                                  <span className="font-bold text-foreground">
                                    Session #{session.id}
                                  </span>
                                  <span
                                    className={`px-2 py-0.5 rounded-full text-[10px] uppercase font-bold tracking-wide ${session.status === 'COMPLETED'
                                      ? "bg-emerald-500/10 text-emerald-600 border border-emerald-500/20"
                                      : session.status === 'IN_PROGRESS'
                                        ? "bg-blue-500/10 text-blue-600 border border-blue-500/20"
                                        : "bg-gray-500/10 text-gray-500"
                                      }`}
                                  >
                                    {session.status}
                                  </span>
                                </div>
                                <p className="text-xs text-muted-foreground">
                                  {new Date(session.started_at).toLocaleString(undefined, {
                                    dateStyle: 'medium',
                                    timeStyle: 'short'
                                  })}
                                </p>
                              </div>

                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => handleViewSessionDetails(session.id)}
                                className="h-8 gap-2 border-ohg-navy/20 text-ohg-navy hover:bg-ohg-navy hover:text-white transition-colors"
                              >
                                {selectedSessionId === session.id ? (
                                  <>Hide Details <ChevronUp className="h-4 w-4" /></>
                                ) : (
                                  <>View Analysis <ChevronDown className="h-4 w-4" /></>
                                )}
                              </Button>
                            </div>

                            {session.topics_list && session.topics_list.length > 0 && (
                              <div className="flex flex-wrap gap-1">
                                {session.topics_list.map((topic: any) => (
                                  <span
                                    key={topic.id}
                                    className="px-2 py-0.5 rounded text-[10px] font-medium bg-background border border-border text-foreground"
                                  >
                                    {topic.name}
                                  </span>
                                ))}
                              </div>
                            )}

                            {session.status === 'COMPLETED' && (
                              <div className="grid grid-cols-2 gap-4 mt-1 bg-background/50 p-3 rounded-lg border border-border/50">
                                <div>
                                  <p className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold mb-0.5">Communication</p>
                                  <p className="text-lg font-bold text-foreground">
                                    {session.communication_score !== null
                                      ? `${(session.communication_score * 100).toFixed(0)}%`
                                      : "-"}
                                  </p>
                                </div>
                                <div>
                                  <p className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold mb-0.5">Technical</p>
                                  <p className="text-lg font-bold text-foreground">
                                    {session.technology_score !== null
                                      ? `${(session.technology_score * 100).toFixed(0)}%`
                                      : "-"}
                                  </p>
                                </div>
                              </div>
                            )}
                          </div>

                          {/* Expanded Details */}
                          {selectedSessionId === session.id && (
                            <div className="border-t border-border bg-muted/30 p-4 animate-in slide-in-from-top-2 duration-300">
                              {isLoadingSessionDetails ? (
                                <div className="text-center py-8 text-muted-foreground flex flex-col items-center">
                                  <RefreshCw className="h-6 w-6 animate-spin mb-2" />
                                  Loading detailed analysis...
                                </div>
                              ) : selectedSessionDetails?.answers && selectedSessionDetails.answers.length > 0 ? (
                                <div className="space-y-4">
                                  <div className="flex justify-between items-center mb-2">
                                    <h4 className="font-bold text-foreground flex items-center gap-2">
                                      <FileText className="h-4 w-4 text-ohg-navy" />
                                      Detailed Q&A Analysis
                                    </h4>
                                  </div>

                                  {selectedSessionDetails.answers.map((answer, index) => {
                                    const accuracyScore = answer.accuracy_score !== null && answer.accuracy_score !== undefined
                                      ? Math.round(answer.accuracy_score * 100)
                                      : answer.topic_score !== null
                                        ? Math.round(answer.topic_score * 100)
                                        : Math.round(answer.similarity_score * 100);

                                    return (
                                      <Card key={answer.id} className="overflow-hidden bg-[#051525] border-border/10 rounded-xl shadow-md relative group">
                                        <div className="p-5">
                                          <div className="flex gap-4">
                                            {/* Number Badge */}
                                            <div className="flex-shrink-0 pt-1">
                                              <div className="w-6 h-6 flex items-center justify-center rounded bg-white/10 text-white/70 font-mono text-xs font-bold border border-white/5">
                                                {index + 1}
                                              </div>
                                            </div>

                                            {/* Content */}
                                            <div className="flex-1 space-y-4">
                                              {/* Question */}
                                              <h4 className="text-base font-bold text-white leading-snug">
                                                {answer.question_text}
                                              </h4>

                                              {/* Answer Box */}
                                              <div className="bg-[#0A2A43]/50 p-4 rounded-lg border border-white/5 relative">
                                                <p className="text-white/60 text-sm leading-relaxed italic font-medium">
                                                  "{answer.user_answer}"
                                                </p>
                                              </div>

                                              {/* Metrics */}
                                              <div className="flex items-center gap-8 pt-2 border-t border-white/5">
                                                <div>
                                                  <div className="text-[10px] text-white/40 font-medium uppercase tracking-wider mb-0.5">Accuracy</div>
                                                  <div className="text-lg font-bold text-ohg-teal">{accuracyScore}%</div>
                                                </div>
                                                <div>
                                                  <div className="text-[10px] text-white/40 font-medium uppercase tracking-wider mb-0.5">Concept Match</div>
                                                  <div className="text-lg font-bold text-white">{Math.round(answer.similarity_score * 100)}%</div>
                                                </div>
                                              </div>
                                            </div>
                                          </div>
                                        </div>
                                      </Card>
                                    );
                                  })}
                                </div>
                              ) : (
                                <div className="text-center py-8 text-muted-foreground">
                                  No answers recorded for this session.
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </Card>
              )}
            </div>
          )}
        </DialogContent>
      </Dialog >
    </div >
  );
};

export default AdminDashboard;
