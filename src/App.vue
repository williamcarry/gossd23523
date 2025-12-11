<script setup lang="ts"></script>

<template>
  <div
    class="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-100 to-slate-200"
  >
    <div class="text-center">
      <!-- TODO: replace everything here with the actual app! -->
      <h1 class="text-2xl font-semibold text-slate-800 flex items-center justify-center gap-3">
        <svg class="animate-spin h-8 w-8 text-slate-400" viewBox="0 0 50 50">
          <circle
            class="opacity-30"
            cx="25"
            cy="25"
            r="20"
            stroke="currentColor"
            stroke-width="5"
            fill="none"
          />
          <circle
            class="text-slate-600"
            cx="25"
            cy="25"
            r="20"
            stroke="currentColor"
            stroke-width="5"
            fill="none"
            stroke-dasharray="100"
            stroke-dashoffset="75"
          />
        </svg>
        Generating your app...
      </h1>
    </div>
  </div>
</template>

<style scoped></style>
