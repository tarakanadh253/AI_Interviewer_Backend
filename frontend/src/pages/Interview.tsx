import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Mic, MicOff, SkipForward, Volume2, Loader2, AlertTriangle, ShieldAlert, Lock, Info, Menu, X } from "lucide-react";
import { cn } from "@/lib/utils";

import { useToast } from "@/hooks/use-toast";
import { apiService, type Question, type InterviewSession } from "@/lib/api";
import { useExamProctoring } from "@/hooks/useExamProctoring";


const Interview = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [session, setSession] = useState<InterviewSession | null>(null);
  const [questions, setQuestions] = useState<Question[]>([]);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [timeLeft, setTimeLeft] = useState(1800); // 30 minutes (will be calculated from session)
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [isInitializingMic, setIsInitializingMic] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [submittedAnswers, setSubmittedAnswers] = useState<Set<number>>(new Set());
  const [speechRecognitionAvailable, setSpeechRecognitionAvailable] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const recognitionRef = useRef<any>(null);
  const synthRef = useRef<SpeechSynthesis | null>(null);
  const transcriptBufferRef = useRef<string>("");
  const isRecordingRef = useRef<boolean>(false);
  const [showMobileQuestions, setShowMobileQuestions] = useState(false);

  const { isFullscreen, isBanned, tabSwitchCount, enterFullscreen } = useExamProctoring({
    isActive: !isLoading && !!session,
    onBan: () => {
      // Stop recording if banned
      if (isRecordingRef.current) {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
          mediaRecorderRef.current.stop();
        }
        if (recognitionRef.current) {
          recognitionRef.current.stop();
        }
        setIsRecording(false);
        isRecordingRef.current = false;
      }
    }
  });

  // Load session and questions on mount
  useEffect(() => {
    const loadInterview = async () => {
      try {
        const sessionId = localStorage.getItem('session_id');
        if (!sessionId) {
          toast({
            title: "No Session Found",
            description: "Please start a new interview.",
            variant: "destructive",
          });
          navigate("/topic-selection");
          return;
        }

        // Get session
        const sessionData = await apiService.getSession(parseInt(sessionId));

        // Check if session is expired or cancelled
        if (sessionData.status === 'CANCELLED') {
          toast({
            title: "Session Expired",
            description: "Your interview session has expired (30 minutes). Please start a new interview.",
            variant: "destructive",
          });
          localStorage.removeItem('session_id');
          navigate("/topic-selection");
          return;
        }

        setSession(sessionData);

        // Calculate remaining time from session start
        if (sessionData.started_at) {
          const startTime = new Date(sessionData.started_at).getTime();
          const now = Date.now();
          const elapsedSeconds = Math.floor((now - startTime) / 1000);
          const totalSeconds = 30 * 60; // 30 minutes
          const remaining = Math.max(0, totalSeconds - elapsedSeconds);
          setTimeLeft(remaining);
        }

        // Get questions for selected topics
        const topicIds = sessionData.topics;
        const allQuestions: Question[] = [];

        // Ensure topicIds is an array
        if (!Array.isArray(topicIds)) {
          console.error('Invalid topics format:', topicIds);
          throw new Error('Invalid session data: topics is not an array');
        }

        for (const topicId of topicIds) {
          const topicQuestions = await apiService.getQuestions(topicId);
          // Ensure topicQuestions is an array before spreading
          if (Array.isArray(topicQuestions)) {
            allQuestions.push(...topicQuestions);
          } else {
            console.warn(`Invalid questions response for topic ${topicId}:`, topicQuestions);
          }
        }

        // Use seeded shuffle based on session ID for consistency when resuming
        // This ensures the same questions appear in the same order
        const seededShuffle = (array: Question[], seed: number) => {
          const shuffled = [...array];
          let currentSeed = seed;
          const random = () => {
            currentSeed = (currentSeed * 9301 + 49297) % 233280;
            return currentSeed / 233280;
          };
          for (let i = shuffled.length - 1; i > 0; i--) {
            const j = Math.floor(random() * (i + 1));
            [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
          }
          return shuffled;
        };

        // Shuffle with session ID as seed for consistency
        const shuffled = seededShuffle(allQuestions, sessionData.id);
        const selectedQuestions = shuffled.slice(0, 10);
        setQuestions(selectedQuestions);

        // Check for existing answers from session data
        // Handle both undefined/null and ensure it's an array
        let answers: any[] = [];
        if (sessionData.answers) {
          if (Array.isArray(sessionData.answers)) {
            answers = sessionData.answers;
          } else {
            console.warn('Invalid answers format (not an array):', sessionData.answers);
            answers = [];
          }
        }

        // Extract answered question IDs
        const answeredQuestionIds = new Set(
          answers
            .map(a => a.question || a.question_id)
            .filter(id => id !== undefined && id !== null)
        );
        setSubmittedAnswers(answeredQuestionIds);

        // Find current question index based on answered questions
        // Set to first unanswered question, or last question if all answered
        if (selectedQuestions.length > 0) {
          const firstUnansweredIndex = selectedQuestions.findIndex(
            q => !answeredQuestionIds.has(q.id)
          );
          if (firstUnansweredIndex !== -1) {
            setCurrentQuestionIndex(firstUnansweredIndex);
          } else {
            // All questions answered, go to last question
            setCurrentQuestionIndex(selectedQuestions.length - 1);
          }
        }

      } catch (error: any) {
        console.error('Error loading interview:', error);
        toast({
          title: "Error",
          description: error.message || "Could not load interview. Please try again.",
          variant: "destructive",
        });
        navigate("/topic-selection");
      } finally {
        setIsLoading(false);
      }
    };

    loadInterview();
  }, [navigate, toast]);

  // Auto-play question audio when question changes or questions are loaded
  useEffect(() => {
    // Only auto-play if fullscreen is active (interview has effectively started)
    if (isFullscreen && questions.length > 0 && currentQuestionIndex >= 0 && currentQuestionIndex < questions.length && !isLoading) {
      // Auto-play question audio when question changes
      // Use setTimeout to ensure browser allows autoplay (user interaction required for first play)
      const timer = setTimeout(() => {
        if (!questions[currentQuestionIndex]) return;

        if (synthRef.current) {
          // Cancel any ongoing speech
          synthRef.current.cancel();

          // Small delay to ensure cancellation completes
          setTimeout(() => {
            const utterance = new SpeechSynthesisUtterance(questions[currentQuestionIndex].question_text);
            utterance.rate = 0.9;
            utterance.pitch = 1;
            utterance.volume = 1;
            utterance.lang = 'en-US';

            utterance.onstart = () => {
              console.debug('Question audio auto-played');
            };

            utterance.onerror = (event) => {
              console.error('Speech synthesis error:', event);
            };

            if (synthRef.current) {
              synthRef.current.speak(utterance);
            }
          }, 100);
        }
      }, 500); // Small delay to ensure UI is ready

      return () => clearTimeout(timer);
    }
  }, [currentQuestionIndex, questions, isLoading, isFullscreen]);

  useEffect(() => {
    if (!session?.started_at) return;

    // Recalculate time left from session start time periodically
    const updateTimeLeft = () => {
      const startTime = new Date(session.started_at).getTime();
      const now = Date.now();
      const elapsedSeconds = Math.floor((now - startTime) / 1000);
      const totalSeconds = 30 * 60; // 30 minutes
      const remaining = Math.max(0, totalSeconds - elapsedSeconds);
      setTimeLeft(remaining);

      if (remaining <= 0) {
        // Auto-complete when time runs out
        if (session) {
          handleComplete();
        }
      }
    };

    // Update immediately
    updateTimeLeft();

    // Update every second
    const timer = setInterval(() => {
      updateTimeLeft();
    }, 1000);

    // Initialize speech synthesis
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      synthRef.current = window.speechSynthesis;
    }

    // Initialize Web Speech API for real-time transcription
    if (typeof window !== 'undefined') {
      const SpeechRecognition = (window as any).webkitSpeechRecognition || (window as any).SpeechRecognition;

      if (SpeechRecognition) {
        try {
          recognitionRef.current = new SpeechRecognition();
          recognitionRef.current.continuous = true;
          recognitionRef.current.interimResults = true;
          recognitionRef.current.lang = 'en-US';
          setSpeechRecognitionAvailable(true);

          recognitionRef.current.onresult = (event: any) => {
            let interimTranscript = '';
            let finalTranscript = transcriptBufferRef.current;

            // Process all results
            for (let i = event.resultIndex; i < event.results.length; i++) {
              const transcript = event.results[i][0].transcript;
              if (event.results[i].isFinal) {
                finalTranscript += transcript + ' ';
              } else {
                interimTranscript += transcript;
              }
            }

            // Update buffer with final transcript
            transcriptBufferRef.current = finalTranscript.trim();

            // Display both final and interim
            setTranscript((finalTranscript + interimTranscript).trim());
          };

          recognitionRef.current.onerror = (event: any) => {
            console.error('Speech recognition error:', event.error);

            let errorMessage = 'Speech recognition error occurred';
            switch (event.error) {
              case 'no-speech':
                errorMessage = 'No speech detected. Please speak clearly.';
                break;
              case 'audio-capture':
                errorMessage = 'Microphone not accessible. Please check permissions.';
                break;
              case 'not-allowed':
                errorMessage = 'Microphone permission denied. Please allow microphone access.';
                break;
              case 'network':
                errorMessage = 'Network error. Please check your connection.';
                break;
              case 'aborted':
                // User stopped, don't show error
                return;
              default:
                errorMessage = `Speech recognition error: ${event.error}`;
            }

            toast({
              title: "Transcription Error",
              description: errorMessage,
              variant: "destructive",
            });

            setIsRecording(false);
            isRecordingRef.current = false;
          };

          recognitionRef.current.onend = () => {
            // If recording was active, restart recognition (for continuous mode)
            if (isRecordingRef.current && recognitionRef.current) {
              try {
                recognitionRef.current.start();
              } catch (err) {
                // Recognition might already be starting
                console.log('Recognition restart:', err);
              }
            }
          };

          recognitionRef.current.onstart = () => {
            console.log('Speech recognition started');
          };
        } catch (error) {
          console.error('Failed to initialize speech recognition:', error);
          setSpeechRecognitionAvailable(false);
          toast({
            title: "Speech Recognition Unavailable",
            description: "Your browser doesn't support speech recognition. You can still type your answers.",
            variant: "destructive",
          });
        }
      } else {
        setSpeechRecognitionAvailable(false);
        console.warn('Speech recognition not available in this browser');
      }
    }

    return () => {
      clearInterval(timer);
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
      if (synthRef.current) {
        synthRef.current.cancel();
      }
    };
  }, [session]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  const playQuestion = () => {
    if (!questions[currentQuestionIndex]) return;

    if (synthRef.current) {
      // Cancel any ongoing speech
      synthRef.current.cancel();

      // Small delay to ensure cancellation completes
      setTimeout(() => {
        const utterance = new SpeechSynthesisUtterance(questions[currentQuestionIndex].question_text);
        utterance.rate = 0.9;
        utterance.pitch = 1;
        utterance.volume = 1;
        utterance.lang = 'en-US';

        utterance.onstart = () => {
          console.debug('Question audio started playing');
        };

        utterance.onerror = (event) => {
          console.error('Speech synthesis error:', event);
        };

        if (synthRef.current) {
          synthRef.current.speak(utterance);
        }
      }, 100);
    }
  };

  const [isSubmitting, setIsSubmitting] = useState(false);

  const submitAnswer = async (answerText: string, specificQuestion?: Question) => {
    // Use specific question if provided, otherwise fallback to current (careful with race conditions)
    const questionToSubmit = specificQuestion || questions[currentQuestionIndex];

    if (!session || !questionToSubmit || !answerText.trim()) {
      return;
    }

    // Don't block UI with loading state for optimistic updates
    // setIsSubmitting(true); 

    // Optimistically mark as submitted immediately
    setSubmittedAnswers(prev => new Set([...prev, questionToSubmit.id]));

    try {
      await apiService.submitAnswer(session.id, questionToSubmit.id, answerText);
      toast({
        title: "Answer recorded",
        description: "Your answer has been saved.",
        duration: 2000,
      });
    } catch (error: any) {
      console.error('Error submitting answer:', error);
      const errorMessage = error.message?.toLowerCase() || "";

      // Handle "Already Answered" error gracefully
      if (errorMessage.includes("already answered") || errorMessage.includes("unique")) {
        console.log("Question already answered, synced with backend.");
      } else {
        // If it failed, maybe we should warn user? 
        // For speed, we rely on the fact that most submissions work.
        // We could add a "retry" queue here in improved version.
        toast({
          title: "Saving failed",
          description: "Could not save your answer. Please try again at the end.",
          variant: "destructive",
        });
        // Revert optimistic update?
        // setSubmittedAnswers(prev => {
        //   const newSet = new Set(prev);
        //   newSet.delete(questionToSubmit.id);
        //   return newSet;
        // });
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleNext = async () => {
    try {
      // capture current state vars before they change
      const currentQ = questions[currentQuestionIndex];
      const currentTranscript = transcript;

      // Stop any ongoing recording
      if (isRecording) {
        stopRecording();
      }

      // Submit current answer if we have transcript - FIRE AND FORGET (Optimistic)
      if (currentTranscript.trim() && session && currentQ) {
        // Check if already submitted to avoid double submission
        if (!submittedAnswers.has(currentQ.id)) {
          // Pass the specific question object to avoid race condition with state index
          submitAnswer(currentTranscript, currentQ);
        }
      }

      // IMMEDIATE UI UPDATE: Move to next question or complete
      if (currentQuestionIndex < questions.length - 1) {
        setCurrentQuestionIndex((prev) => prev + 1);
        setTranscript("");
        transcriptBufferRef.current = ""; // Reset transcript buffer
        setIsRecording(false);
        // Question audio will auto-play via useEffect when currentQuestionIndex changes
      } else {
        await handleComplete();
      }
    } catch (error) {
      console.error("Error in handleNext:", error);
      toast({
        title: "Navigation Error",
        description: "Could not proceed to next question. Please try again.",
        variant: "destructive"
      });
    }
  };

  const handleComplete = async () => {
    if (!session) return;

    // Optimistically navigate immediately
    navigate("/results");

    try {
      // Fire and forget - complete session in background
      apiService.completeSession(session.id)
        .then(completedSession => {
          localStorage.setItem('completed_session_id', completedSession.id.toString());
        })
        .catch(error => {
          console.error('Error completing session in background:', error);
          // Even if it fails, the results page load might fix it or we can handle it there
        });

    } catch (error: any) {
      console.error('Error firing completion:', error);
    }
  };

  const startRecording = async () => {
    setIsInitializingMic(true);
    try {
      // Request microphone permission
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        // Audio is recorded but transcription is handled by Web Speech API
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();

      // Start speech recognition if available
      if (recognitionRef.current && speechRecognitionAvailable) {
        try {
          // Reset transcript buffer
          transcriptBufferRef.current = "";
          setTranscript("");

          recognitionRef.current.start();
          console.log('Speech recognition started');
        } catch (err: any) {
          console.log('Speech recognition start error:', err);
          if (err.name !== 'InvalidStateError') {
            toast({
              title: "Transcription Error",
              description: "Could not start speech recognition. You can still type your answer.",
              variant: "destructive",
              duration: 3000,
            });
          }
        }
      } else if (!speechRecognitionAvailable) {
        toast({
          title: "Speech Recognition Unavailable",
          description: "Your browser doesn't support speech recognition. Please type your answer manually.",
          variant: "destructive",
          duration: 3000,
        });
      }

      setIsRecording(true);
      isRecordingRef.current = true;
    } catch (error: any) {
      console.error('Error starting recording:', error);
      let errorMessage = "Could not access microphone";

      if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
        errorMessage = "Microphone permission denied. Please allow microphone access in your browser settings.";
      } else if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
        errorMessage = "No microphone found. Please connect a microphone and try again.";
      } else if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
        errorMessage = "Microphone is being used by another application. Please close other apps using the microphone.";
      }

      toast({
        title: "Recording failed",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setIsInitializingMic(false);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      if (mediaRecorderRef.current.state !== 'inactive') {
        try {
          mediaRecorderRef.current.stop();
        } catch (error: any) {
          console.error('Error stopping recorder:', error);
        }
      }

      if (recognitionRef.current) {
        try {
          recognitionRef.current.stop();
          console.log('Speech recognition stopped');
        } catch (err) {
          console.log('Speech recognition stop error:', err);
        }
      }

      setIsRecording(false);
      isRecordingRef.current = false;
      setIsTranscribing(false);

      // Finalize transcript from buffer
      if (transcriptBufferRef.current) {
        setTranscript(transcriptBufferRef.current);
      }
    }
  };

  // Transcription is handled entirely by Web Speech API
  // No need for Supabase or external services

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4 text-primary" />
          <p className="text-muted-foreground">Loading interview...</p>
        </div>
      </div>
    );
  }

  if (!questions.length) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Card className="p-8">
          <p className="text-muted-foreground">No questions available. Please select courses and try again.</p>
          <Button onClick={() => navigate("/topic-selection")} className="mt-4">
            Go Back
          </Button>
        </Card>
      </div>
    );
  }


  const currentQuestion = questions[currentQuestionIndex];
  const isAnswered = currentQuestion && submittedAnswers.has(currentQuestion.id);

  // Banned State Overlay
  if (isBanned) {
    return (
      <div className="fixed inset-0 z-50 bg-background/95 backdrop-blur-sm flex flex-col items-center justify-center p-4 text-center animate-in fade-in duration-500 font-sans">
        <div className="max-w-lg w-full bg-card p-10 rounded-3xl border border-destructive/20 shadow-2xl">
          <div className="w-20 h-20 rounded-full bg-destructive/10 flex items-center justify-center mx-auto mb-8 animate-[pulse_3s_ease-in-out_infinite]">
            <Lock className="w-10 h-10 text-destructive" />
          </div>
          <h1 className="text-3xl font-bold text-foreground mb-4 tracking-tight">Interview Terminated</h1>
          <p className="text-muted-foreground mb-8 leading-relaxed">
            This session has been permanently locked due to multiple violations of the examination integrity rules.
          </p>

          <div className="flex flex-col gap-3 mb-8">
            <div className="bg-muted/50 p-4 rounded-xl border border-border flex items-center justify-between">
              <span className="text-muted-foreground text-sm">Violation Type</span>
              <span className="text-destructive font-mono text-sm font-semibold">TAB_SWITCH / FOCUS_LOST</span>
            </div>
            <div className="bg-muted/50 p-4 rounded-xl border border-border flex items-center justify-between">
              <span className="text-muted-foreground text-sm">Total Violations</span>
              <span className="text-destructive font-mono text-sm font-semibold">{tabSwitchCount} / 2 Allowed</span>
            </div>
          </div>

          <Button
            onClick={() => navigate('/')}
            variant="outline"
            className="w-full border-border hover:bg-muted text-foreground h-12 text-base font-medium"
          >
            Return to Dashboard
          </Button>
        </div>
      </div>
    );
  }

  // Fullscreen Enforcement Overlay
  if (!isFullscreen && !isLoading && questions.length > 0) {
    return (
      <div className="fixed inset-0 z-50 bg-background/95 backdrop-blur-md flex flex-col items-center justify-center p-4 animate-in fade-in duration-300">
        <Card className="max-w-md w-full p-8 bg-card border-border shadow-xl relative overflow-hidden">
          {/* Decorative background vibe */}
          <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-ohg-navy to-ohg-teal" />

          <div className="w-16 h-16 rounded-2xl bg-ohg-teal/10 flex items-center justify-center mx-auto mb-6 shadow-sm">
            <ShieldAlert className="w-8 h-8 text-ohg-teal" />
          </div>

          <h2 className="text-2xl font-bold text-center text-foreground mb-3">Secure Environment</h2>
          <p className="text-center text-muted-foreground mb-8 leading-relaxed text-sm">
            To ensure interview integrity, this session requires <strong>Fullscreen Mode</strong>.
            Exiting fullscreen or switching tabs will be recorded as a violation.
          </p>

          <div className="space-y-4">
            <Button
              onClick={async () => {
                await enterFullscreen();
                // We don't need to manually play audio here because the useEffect keeps track of isFullscreen
                // changing to true, and triggers the audio.
                // However, to be doubly sure and immediate:
                // playQuestion(); // This might conflict with the useEffect.
              }}
              className="w-full py-6 text-base bg-ohg-navy hover:bg-ohg-navy/90 text-white shadow-lg shadow-ohg-navy/20 transition-all hover:scale-[1.02]"
            >
              Enter Fullscreen & Start
            </Button>

            <p className="text-[10px] text-center text-gray-400 uppercase tracking-widest font-semibold">
              Violations Recorded: {tabSwitchCount} / 2
            </p>
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col lg:flex-row bg-background">
      {/* Sidebar - Brand Navy */}
      <div className="w-full lg:w-80 bg-ohg-navy border-b lg:border-b-0 lg:border-r border-ohg-navy flex flex-col lg:sticky lg:top-0 lg:h-screen z-20 text-white shadow-2xl">

        {/* Header */}
        <div className="p-8 border-b border-white/10 bg-ohg-navy">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
              <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-ohg-orange text-white">
                <span className="font-bold">Q</span>
              </span>
              Overview
            </h2>
            <Button
              variant="ghost"
              size="icon"
              className="lg:hidden text-white/70 hover:text-white"
              onClick={() => setShowMobileQuestions(!showMobileQuestions)}
            >
              {showMobileQuestions ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </Button>
          </div>

          <div className={cn("mt-6", !showMobileQuestions && "hidden lg:block")}>
            <div className="flex justify-between text-xs font-medium text-white/60 mb-2">
              <span>Progress</span>
              <span>{questions.length > 0 ? Math.round((submittedAnswers.size / questions.length) * 100) : 0}%</span>
            </div>
            <div className="h-1.5 w-full bg-white/10 rounded-full overflow-hidden">
              <div
                className="h-full bg-ohg-orange transition-all duration-500 ease-out rounded-full shadow-[0_0_10px_rgba(242,97,35,0.5)]"
                style={{ width: `${questions.length > 0 ? (submittedAnswers.size / questions.length) * 100 : 0}%` }}
              />
            </div>
          </div>
        </div>

        {/* Questions List */}
        <div className={cn("flex-1 overflow-y-auto py-4 px-4 custom-scrollbar-light", !showMobileQuestions && "hidden lg:block")}>
          <div className="space-y-1">
            {questions.map((q, index) => {
              const isCurrent = index === currentQuestionIndex;
              const isAnswered = submittedAnswers.has(q.id);

              return (
                <div
                  key={q.id}
                  className={`group relative mb-2 rounded-xl border p-4 transition-all duration-300 ${isCurrent
                    ? "bg-white/10 border-ohg-orange/50 shadow-inner"
                    : isAnswered
                      ? "bg-ohg-teal/10 border-transparent hover:bg-white/5"
                      : "bg-transparent border-transparent hover:bg-white/5"
                    }`}
                >
                  {/* Active Indicator Line */}
                  {isCurrent && (
                    <div className="absolute left-0 top-2 bottom-2 w-1 rounded-r-full bg-ohg-orange shadow-[0_0_10px_currentColor]" />
                  )}

                  <div className="flex items-start gap-4 pl-2">
                    {/* Step Number/Icon */}
                    <div className="flex-shrink-0 pt-0.5">
                      <div
                        className={`flex h-8 w-8 items-center justify-center rounded-lg border text-sm font-bold transition-all duration-300 ${isCurrent
                          ? "border-ohg-orange bg-ohg-orange text-white shadow-lg scale-110"
                          : isAnswered
                            ? "border-ohg-teal bg-ohg-teal/20 text-ohg-teal"
                            : "border-white/10 bg-white/5 text-white/40"
                          }`}
                      >
                        {isAnswered ? "✓" : index + 1}
                      </div>
                    </div>

                    {/* Content */}
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center justify-between mb-0.5">
                        <span className={`text-sm font-semibold truncate pr-2 ${isCurrent ? "text-white" : isAnswered ? "text-ohg-teal" : "text-white/40"
                          }`}>
                          Question {index + 1}
                        </span>
                        {isCurrent && (
                          <span className="flex h-2 w-2">
                            <span className="animate-ping absolute inline-flex h-2 w-2 rounded-full bg-ohg-orange opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-2 w-2 bg-ohg-orange"></span>
                          </span>
                        )}
                      </div>

                      <div className="flex items-center gap-2">
                        <span className={`text-xs ${isCurrent ? "text-white/60" : "text-white/30"}`}>
                          {currentQuestionIndex === index ? "Active" : isAnswered ? "Completed" : "Pending"}
                        </span>
                        {/* Difficulty Dot */}
                        <span className="h-1 w-1 rounded-full bg-white/20"></span>
                        <span className="text-xs text-white/30 uppercase tracking-wider">{q.difficulty}</span>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Footer */}
        <div className="p-6 border-t border-white/10 bg-ohg-navy text-center">
          <p className="text-xs text-white/40 font-medium">
            Step {currentQuestionIndex + 1} of {questions.length}
          </p>
        </div>
      </div>

      {/* Main Content - Light & Clean */}
      <div className="flex-1 overflow-y-auto min-h-screen bg-background">
        <div className="max-w-5xl mx-auto p-4 md:p-8 lg:p-12">

          {/* Timer & Meta */}
          <div className="flex justify-between items-end mb-8 border-b border-border pb-4">
            <div>
              <span className="text-xs font-bold text-ohg-orange tracking-widest uppercase mb-1 block">
                Current Question
              </span>
              <div className="text-sm text-foreground/80">
                {currentQuestion && (
                  <span className="flex items-center gap-2">
                    {currentQuestion.topic_name}
                    <span className="h-1 w-1 rounded-full bg-muted-foreground/30"></span>
                    <span className="uppercase text-[10px] tracking-wider border border-border px-1.5 py-0.5 rounded text-muted-foreground bg-card">
                      {currentQuestion.difficulty}
                    </span>
                    {tabSwitchCount > 0 && (
                      <span className="flex items-center gap-1 text-amber-600 bg-amber-50 px-2 py-0.5 rounded text-[10px] border border-amber-200 ml-2 font-mono">
                        <AlertTriangle className="w-3 h-3" />
                        WARNINGS: {tabSwitchCount}/2
                      </span>
                    )}
                  </span>
                )}
              </div>
            </div>
            <div className={`font-mono text-4xl font-light tracking-tighter ${timeLeft < 300 ? "text-red-500 animate-pulse" : "text-ohg-navy"}`}>
              {formatTime(timeLeft)}
            </div>
          </div>

          {/* Question Display */}
          <div className="mb-12 text-center space-y-8 animate-fade-in-up">
            <h2
              className="text-3xl md:text-4xl lg:text-5xl font-bold leading-tight text-foreground select-none cursor-default"
              onContextMenu={(e) => e.preventDefault()}
              onCopy={(e) => { e.preventDefault(); return false; }}
            >
              {currentQuestion?.question_text}
            </h2>

            <div className="flex justify-center">
              <Button
                variant="ghost"
                size="sm"
                className="text-gray-500 hover:text-ohg-navy hover:bg-gray-100 rounded-full px-6 border border-transparent hover:border-gray-200 transition-all font-medium"
                onClick={playQuestion}
              >
                <Volume2 className="h-4 w-4 mr-2 text-ohg-teal" />
                Replay Audio
              </Button>
            </div>
          </div>

          {/* Recording / Interaction Area */}
          <div className="relative max-w-2xl mx-auto">
            <Card className="relative p-1 bg-card border-border rounded-[2rem] shadow-xl overflow-hidden">
              <div className="bg-muted/30 rounded-[1.8rem] p-8 md:p-10 flex flex-col items-center gap-6 relative overflow-hidden">
                {/* Ambience inside card */}
                <div className="absolute top-0 right-0 w-64 h-64 bg-ohg-teal/5 rounded-full blur-3xl pointer-events-none -translate-y-1/2 translate-x-1/2" />
                <div className="absolute bottom-0 left-0 w-64 h-64 bg-ohg-orange/5 rounded-full blur-3xl pointer-events-none translate-y-1/2 -translate-x-1/2" />

                {/* Visualizer / Status */}
                <div className="h-12 flex items-center justify-center gap-1 w-full relative z-10">
                  {isRecording ? (
                    <div className="flex items-center gap-1 h-12">
                      {[...Array(8)].map((_, i) => (
                        <div
                          key={i}
                          className="w-1.5 bg-ohg-orange rounded-full animate-music-bar"
                          style={{
                            animationDelay: `${i * 0.1}s`,
                          }}
                        />
                      ))}
                    </div>
                  ) : (
                    <span className="text-sm font-semibold text-muted-foreground uppercase tracking-widest">
                      {isAnswered ? "Answer Recorded" : "Ready to Record"}
                    </span>
                  )}
                </div>

                {/* Big Mic Button */}
                <div className="relative group z-10">
                  {isRecording && (
                    <div className="absolute inset-0 bg-ohg-orange/20 rounded-full animate-ping"></div>
                  )}
                  <Button
                    onClick={toggleRecording}
                    disabled={isTranscribing || isAnswered || isInitializingMic}
                    className={`relative w-24 h-24 rounded-full flex items-center justify-center transition-all duration-500 border-4 ${(isRecording || isInitializingMic)
                      ? "bg-white border-red-500 text-red-500 shadow-xl"
                      : isAnswered
                        ? "bg-emerald-500/10 border-emerald-500 text-emerald-500 cursor-default"
                        : "bg-card border-border text-muted-foreground hover:border-ohg-orange hover:text-white hover:bg-ohg-orange hover:scale-105 hover:shadow-lg shadow-sm"
                      }`}
                  >
                    {isInitializingMic ? (
                      <Loader2 className="h-10 w-10 animate-spin text-red-500" />
                    ) : isRecording ? (
                      <div className="w-8 h-8 rounded bg-red-500 animate-pulse shadow-sm" />
                    ) : isAnswered ? (
                      <span className="text-2xl font-bold">✓</span>
                    ) : (
                      <Mic className="h-10 w-10" />
                    )}
                  </Button>
                </div>

                <p className="text-sm font-medium text-gray-500 relative z-10">
                  {isTranscribing
                    ? "Transcribing your answer..."
                    : isRecording
                      ? "Listening... Tap to stop"
                      : isAnswered
                        ? "Great job! Answer submitted."
                        : "Tap the microphone to start"
                  }
                </p>

                {/* Transcript Area */}
                {(transcript || isRecording) && (
                  <div className="w-full mt-4 p-4 rounded-xl bg-card border border-border text-left max-h-40 overflow-y-auto custom-scrollbar-light shadow-sm relative z-10">
                    {isRecording && !transcript && (
                      <div className="flex items-center gap-2 text-red-500 text-xs font-bold uppercase tracking-wider mb-2">
                        <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></span>
                        Live
                      </div>
                    )}
                    <p className="text-foreground leading-relaxed">
                      {transcript || (speechRecognitionAvailable ? "Speak clearly..." : "Listening...")}
                    </p>
                  </div>
                )}
              </div>
            </Card>
          </div>

          {/* Navigation */}
          <div className="mt-12 flex justify-center">
            <Button
              onClick={handleNext}
              disabled={isTranscribing || isSubmitting}
              className={`px-10 py-6 text-lg rounded-full font-bold tracking-wide transition-all duration-300 select-none ${(isAnswered || transcript)
                ? "bg-ohg-navy text-white hover:bg-[#051525] shadow-xl hover:scale-105" // High contrast "Next"
                : "bg-gray-200 text-gray-400 hover:bg-gray-300"
                }`}
            >
              {isSubmitting ? (
                <span className="flex items-center gap-3">
                  <Loader2 className="h-5 w-5 animate-spin" /> Processing...
                </span>
              ) : currentQuestionIndex < questions.length - 1 ? (
                <span className="flex items-center gap-3">
                  Next Question <SkipForward className="h-5 w-5" />
                </span>
              ) : (
                "Finish Interview"
              )}
            </Button>
          </div>

          {/* Footer Tip */}
          <div className="mt-12 text-center opacity-40 hover:opacity-100 transition-opacity">
            <p className="text-xs font-bold text-muted-foreground uppercase tracking-[0.2em]">
              Speak naturally • Take your time
            </p>
          </div>

        </div>
      </div>
    </div>
  );
};

export default Interview;
