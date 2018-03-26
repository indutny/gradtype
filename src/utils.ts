export function shuffle(list: any[]): void {
  for (let i = 0; i < list.length - 1; i++) {
    const j = i + 1 + Math.floor(Math.random() * (list.length - 1 - i));

    const t = list[i];
    list[i] = list[j];
    list[j] = t;
  }
}
