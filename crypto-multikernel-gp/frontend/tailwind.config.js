/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#0D1418",
        panel: "#141C22",
        "panel-raised": "#1B252C",
        "text-primary": "#E8EDEF",
        "text-secondary": "#8A9BA5",
        fundamental: "#4FA8A0",
        technical: "#E0A458",
        sentiment: "#C46B8A",
        interaction: "#566270",
        signature: "#F2C879",
        gridline: "#232F38",
      },
      fontFamily: {
        display: ["Space Grotesk", "sans-serif"],
        body: ["IBM Plex Sans", "sans-serif"],
        mono: ["IBM Plex Mono", "monospace"],
      },
    },
  },
  plugins: [],
};
