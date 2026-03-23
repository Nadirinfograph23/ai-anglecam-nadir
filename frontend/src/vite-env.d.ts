/// <reference types="vite/client" />

interface PuterAI {
  chat(
    prompt: string,
    options?: { model?: string; stream?: boolean }
  ): Promise<string>;
  chat(
    prompt: string,
    imageUrl: string,
    options?: { model?: string }
  ): Promise<string>;
  txt2img(
    prompt: string,
    testMode?: boolean
  ): Promise<HTMLImageElement>;
  txt2img(
    options: {
      prompt: string;
      model?: string;
      provider?: string;
      quality?: string;
      ratio?: { w: number; h: number };
      test_mode?: boolean;
    }
  ): Promise<HTMLImageElement>;
}

interface Puter {
  ai: PuterAI;
  print(text: string, options?: { code?: boolean }): void;
}

declare const puter: Puter;
