const sentences = [
  "He turned round and, walking to the window, drew up the blind",
  "The true mystery of the world is the visible, not the invisible",
  "The common people who acted with me seemed to me to be godlike",
  "Having locked the door behind him, he crept quietly downstairs",
  "Success was given to the strong, failure thrust upon the weak",
  "I should not be sorry to see you disgraced, publicly disgraced",
  "England is bad enough I know, and English society is all wrong",
  "And I don't think it really matters about your not being there",
  "It was the imagination that set remorse to dog the feet of sin",
  "I was away with my love in a forest that no man had ever seen",
  "As for being poisoned by a book, there is no such thing as that",
  "The folk don't like to have that sort of thing in their houses",
  "Modern morality consists in accepting the standard of one's age",
  "The flower seemed to quiver, and then swayed gently to and fro",
  "Every month as it wanes brings you nearer to something dreadful",
  "She will represent something to you that you have never known",
  "Our grandmothers painted in order to try and talk brilliantly",
  "It would have made me in love with love for the rest of my life",
  "It is the confession, not the priest, that gives us absolution",
  "I want you to get rid of the dreadful people you associate with"
];

export default sentences.map((sentence) => {
  return sentence.replace(/\s/g, '␣').toLowerCase();
});
