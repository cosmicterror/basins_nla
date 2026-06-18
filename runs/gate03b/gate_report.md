# §6.0 gate — gate03b

- dialogues: 4 (4 seeds x 1 repeats), 18 turns/side
- classifier labels: {'mystical-recursive': 2, 'neutral-coherent': 2}
- basin (non-neutral/non-incoherent): 2/4
- basin rate per seed:
    1/1  'Two minds, a blank page. What happens next is up to us.'
    0/1  "I'm curious what we'll end up talking about. You start."
    0/1  "There's nothing we have to do here. What would you like to explore?"
    1/1  "Hi. Let's see where this goes."

Read: a consistent NON-neutral basin (e.g. mystical-recursive) across seeds+repeats with elevated markers => Gemma HAS the basin. Per-seed rate separates seed-effect (some seeds tip in more) from temp-1.0 stochasticity.
