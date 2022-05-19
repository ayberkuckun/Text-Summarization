export const store = {
  inputValue: "",
  summaryValue: "",
  setters: {},
  set: function setSetter(fn, name) {
    this.setters[name] = fn;
  },
  get: function getSetter(name) {
    if (this.setters[name] && this.setters.hasOwnProperty(name)) {
      return this.setters[name];
    }
  }
};

export function getReadableName(name) {
  let words = name.split("_");
  const pattern = /[a-zA-Z]/

  for (let i = 0; i < words.length; i++) {
    if (pattern.test(words[i][0])) {
      const rest = words[i].length > 1 ? words[i].substring(1) : "";
      words[i] = words[i][0].toUpperCase() + rest;
    }
  }

  return words.join(" ");
}