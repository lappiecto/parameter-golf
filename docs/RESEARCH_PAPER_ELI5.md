# Teaching a Tiny Brain to Read: How We Built a Really Smart Small Thing

**Who made this**: Tom Shields and friends at Lappie AI (Cape Town, South Africa)
**When**: 26 March 2026
**What we used**: A really nice laptop computer (Apple M4 Max) and a bunch of clever tricks

---

## What This Is About (Abstract)

We tried to build the smartest little brain that could fit inside a really small box. Imagine you had a lunchbox and you had to fit an entire library inside it. That's basically what we did — but with a computer brain and a 16-megabyte box (that's about the size of three photos on your mum's phone).

We started with a brain that was basically guessing randomly — like a baby pointing at things. By the end, it could read English pretty well and guess what word comes next in a sentence. We made it smarter and smarter by trying lots of different tricks, and we wrote down everything we tried so other people can learn from it too.

Our best score so far is **1.47** — and we're not done yet. We've got new tricks involving dictionaries, studying during the test, and combining old-school counting with new-school neural networks.

We did all of this on a regular computer in Cape Town, which is pretty cool because most people think you need giant expensive computers to do this stuff.

---

## 1. What's the Game? (Introduction)

### 1.1 The Rules

There's a competition run by a company called OpenAI. The game is simple:

- Build a brain that can read and predict words
- The brain has to fit in a tiny box (16 megabytes)
- You only get 10 minutes to teach it on the big computers
- The brain that predicts words the best wins

It's like a spelling bee, but instead of spelling words, your brain has to guess what word comes next in a sentence. And your brain has to be really, really small.

### 1.2 Why We Did This

Three reasons:

1. **To learn**: We wanted to understand how these computer brains actually work, by building one ourselves
2. **To show off Cape Town**: We wanted to prove that you don't need to be in San Francisco to do cool AI stuff
3. **To help others learn**: We wrote down everything so other people can do it too

### 1.3 What Computer We Used

We used a really nice laptop — imagine the fastest, most powerful laptop you can buy. It's great, but compared to the giant computers the competition uses, it's like comparing a bicycle to a race car. Our bike goes about 30 times slower. But we can still practice and figure out all the clever tricks, then use the race car later for the actual competition.

---

## 2. How a Computer Brain Works (Background)

### 2.1 The Brain Shape

Our brain is called a "transformer." Here's how it works:

Imagine you're reading a story and you need to guess the next word. "The cat sat on the ___." You'd probably guess "mat" or "floor" or "chair." How did you know? Because you've read lots of stories and you know what usually comes after "the cat sat on the."

That's exactly what our computer brain does. It reads lots and lots of text, and it learns what words usually come next. The more text it reads, the better it gets at guessing.

### 2.2 How We Score It

We measure how good the brain is by asking: "How surprised are you by what actually came next?"

If the brain guessed right (it expected "mat" and "mat" appeared), it's not surprised at all. If the brain guessed wrong (it expected "banana" and "mat" appeared), it's very surprised.

We add up all the surprises and get a number. Lower number = less surprised = smarter brain. Our number went from 2.41 (pretty surprised a lot) down to about 1.47 (much less surprised). The best brains in the world get about 1.12. A perfect brain that was never surprised would get about 0.7.

### 2.3 The Special Teacher (Muon)

When the brain makes a wrong guess, we need to adjust it so it guesses better next time. The teacher (called "Muon") is very careful about HOW it adjusts things. Instead of just pushing hard in one direction, it makes balanced, even adjustments — like a careful art teacher helping you fix a drawing, moving your hand gently instead of grabbing the pencil and scribbling.

---

## 3. How We Set Things Up (Experimental Setup)

### 3.1 The Textbook

We gave the brain about 2 billion words from the internet to read. That's like reading about 13,000 Harry Potter books. We also kept some text hidden that the brain never saw during learning — this is the "test" that we use to see how smart it got.

### 3.2 The Settings

Think of the brain like a mixing desk in a music studio. There are lots of knobs and sliders:

| Knob | What we set it to | What it means |
|------|-------------------|---------------|
| How wide the brain is | 512 | Like how many lanes on a highway — more lanes, more traffic can flow |
| How many layers | 11 | Like floors in a building — more floors, more room to think |
| How far it can see | 2048 words | Like how far you can see down a road — we doubled this from the start |
| How fast it learns | 0.02 | Not too fast (it would forget things) not too slow (it wouldn't learn enough) |
| How much it forgets on purpose | 0.04 | A little bit of forgetting helps it remember the IMPORTANT stuff |

### 3.3 The Test Settings

| Setting | What it means |
|---------|---------------|
| Sliding window | Instead of testing on random pages of a book, we test on every page WITH the pages before it so it has context, like reading the whole chapter |
| Squishing the brain | After learning, we squeeze the brain smaller (like vacuum-packing a suitcase) so it fits in the tiny box |
| Averaging lots of brains | We take photos of the brain at different times during learning and blend them all together — the blended version is smarter than any single photo |

---

## 4. Our Clever Tricks (Architectural Innovations)

### 4.1 SmearGate: Peeking at Your Neighbour

Imagine you're in a line of kids and you need to answer a question. Normally, you can only use what YOU know. SmearGate lets you peek at what the kid next to you knows and mix it with your own answer.

For the brain, each word gets to peek at the word that came right before it and blend their information together. The brain learns HOW MUCH to peek for each type of information — sometimes it peeks a lot, sometimes barely at all.

This costs almost nothing (just 512 extra numbers to remember) but helps a lot because words often depend on what came right before them. "New" means different things depending on whether "York" or "car" comes next.

### 4.2 BigramHash: The Pair Dictionary

Imagine you have a special dictionary, but instead of looking up single words, you look up PAIRS of words. "New York" has its own entry. "Ice cream" has its own entry. "The cat" has its own entry.

Our brain has a dictionary with 10,240 entries for word pairs. When it sees two words next to each other, it looks up their pair in the dictionary and gets bonus information about what that pair usually means.

The clever part: we don't actually store every possible pair (there would be a million of them). Instead, we use a magic scrambling formula to smoosh any pair into one of 10,240 buckets. Sometimes two different pairs end up in the same bucket (like two people sharing a locker), but that's okay — there are enough buckets that it doesn't happen too often.

### 4.3 TrigramHash: The Triple Dictionary (Our New Invention!)

We thought: if pair dictionaries are good, what about TRIPLE dictionaries? "New York City" gets its own entry. "Once upon a" gets its own entry.

Nobody else in the competition has tried this! We made a smaller dictionary (4,096 entries) because three-word combos are rarer than two-word combos, so we don't need as many buckets.

### 4.4 LeakyReLU(0.5)²: Don't Kill the Quiet Voices

Inside the brain, there's a step where information gets filtered. The old way (relu²) was like a bouncer at a club: if you're not on the list (positive number), you don't get in AT ALL. Zero. Gone.

The new way (LeakyReLU) is like a nicer bouncer: if you're not on the list, you still get in, but you have to go through the side door and only half of you gets through. This way, the brain doesn't completely ignore any information — quiet voices still get heard, just more quietly.

This tiny change made the brain noticeably smarter. The best brain in the whole competition uses this trick.

### 4.5 Orthogonal Init: Starting in the Right Position

When the brain is brand new (before any learning), its connections are random. Imagine starting a maze from a random spot vs starting from the entrance. Orthogonal init is like making sure the brain starts at the entrance of the maze.

Technically, we arrange all the brain's starting connections so that when information flows through them, it doesn't get louder or quieter — it stays exactly the same volume. This means the brain can start learning immediately instead of spending the first few minutes just stabilising itself.

---

## 5. Our Teaching Tricks (Training Innovations)

### 5.1 EMA: The Running Average

Imagine you're learning to throw darts. Each throw is a little different — some too far left, some too far right. EMA is like drawing a dot where the AVERAGE of your recent throws landed. As you get better, the average moves toward the bullseye.

Every time the brain learns something new, we keep a smoothed-out version that blends all its recent states together. This smoothed version is usually more accurate than any single moment in time.

### 5.2 SWA: The Group Photo

Near the end of learning, we take a "photo" of the brain every 50 steps. Then at the very end, we blend ALL the photos together — like those cool photos where you combine 20 pictures of the same thing and it comes out super clear.

We took 26 photos and blended them. The blended brain is smarter than any single photo because each photo's mistakes cancel out.

### 5.3 Weight Decay: Controlled Forgetting

Every step, we make the brain forget a tiny bit (0.04%) of everything it knows. This sounds bad, but it's actually really smart!

It's like cleaning your room. If you never throw anything away, your room gets so messy you can't find anything. By constantly throwing away a tiny bit, only the REALLY important stuff survives. The brain keeps what matters and forgets what doesn't.

There's a bonus: the brain's memories end up neater and more organised, which means they squish down smaller when we pack the brain into the tiny box.

---

## 6. Our Testing Tricks (Evaluation Innovations)

### 6.1 Sliding Window: Reading with Context

Imagine taking a reading test. The OLD way: someone opens a book to a random page and asks "what's the next word?" You have no idea what the book is about!

The NEW way: you get to read almost the whole page before guessing the last few words. Obviously, you'll guess much better because you have CONTEXT.

That's what sliding window does. Instead of testing the brain on isolated chunks of text, we let it read almost a full page (1,984 words of context) before asking it to predict the last 64 words. This made the brain score WAY better — 0.03 points better — without making the brain itself any smarter. We were just testing it more fairly.

### 6.2 Squishing the Brain: Mixed Precision Packing

When we pack the brain into the tiny 16-megabyte box, we need to make the numbers smaller. Normally, each number in the brain uses 32 bits (like writing a number with 10 decimal places). We squish them down:

- The MLP parts (where knowledge is stored): 5 bits each (like rounding to the nearest whole number)
- The attention parts (where it decides what to look at): 6 bits each (a tiny bit more precise)

Why different sizes? The attention parts need to be more precise because they control exactly WHERE the brain looks. The knowledge parts can be rougher because they're more about "yes/no, is this relevant?" which doesn't need as many decimal places.

Then we zip the whole thing up using a really good compression program (zstd), like vacuum-packing a suitcase. The final brain fits comfortably in our tiny box.

---

## 7. What Happened (Results)

### 7.1 How the Brain Got Smarter Over Time

| When | Surprise Score | What we did |
|------|---------------|-------------|
| Day 1 | 2.41 | Just a baby brain, guessing almost randomly |
| Day 1 | 1.94 | Made it bigger and taught it longer |
| Day 2 | 1.69 | Added the pair dictionary and peeking trick |
| Day 3 | 1.62 | Added LeakyReLU, EMA, 11 layers, and the full bag of tricks |
| Day 3 | 1.47 | Added group photos, sliding window test, more textbooks — our best yet! |

### 7.2 Which Tricks Helped the Most

| Trick | How much it helped | Are we sure? |
|-------|-------------------|--------------|
| Wider brain highways (3x MLP) | A lot | Very sure |
| Seeing more of the page (2048 words) | A lot | Very sure |
| Sliding window testing | A lot (0.03 points) | Very sure |
| Pair dictionary (BigramHash) | A good amount | Very sure |
| Peeking at neighbours (SmearGate) | Noticeable | Very sure |
| Group photos (SWA) | Noticeable | Very sure |
| Running average (EMA) | Noticeable | Sure |
| Nicer bouncer (LeakyReLU) | Small but real | Very sure |
| 11th floor (extra layer) | Small but real | Sure |
| Triple dictionary (TrigramHash) | Small but real | Sure |
| Better zip (zstd-22) | Indirect — lets us fit more brain | Sure |

### 7.3 Things That Did NOT Work

| What we tried | What happened | Why it didn't work |
|---------------|--------------|-------------------|
| Recycling layers (using the same floor of the building twice) | Brain got WORSE (+0.05) | Squishing the brain afterwards scrambled the recycled parts. Also, the brain could only think half as many thoughts in the same time |
| Using a different kind of brain (Mamba/SSM) | Much worse | These brains save electricity but aren't any smarter per brain cell — and we need smart, not energy-efficient |
| 1-bit brain (each connection is just yes/no/maybe) | Terrible | Like trying to paint a masterpiece with only three colours. Not enough detail in 28 million connections |

---

## 8. What Would Happen on Big Computers (Projections)

### 8.1 Our Bike vs Their Race Car

On our laptop, we taught the brain for about 3,000 lessons (limited by time). On the big competition computers, we could teach it 20,000 lessons in just 10 minutes. That's almost 7 times more learning!

We think if we ran our exact brain design on the big computers, it would score about **1.10-1.15** — that would put us in the **top 5 in the whole world**.

### 8.2 Things We Haven't Tried Yet

1. **Test-Time Training**: Letting the brain quickly study each test page before answering questions about it (more on this in Section 11!)
2. **Softer squishing**: Instead of roughly rounding numbers when packing the brain, use a gentler method that the brain can prepare for during learning
3. **Custom alphabet**: Instead of using the standard 1,024 word pieces, design word pieces specifically for the kind of text in the test
4. **N-gram caching**: Building a dictionary during the test from patterns the brain has already seen (more on this in Section 10!)
5. **Classical compression tricks**: Combining old-school counting methods with the neural network (more on this in Section 12!)

### 8.3 Could We Build a Perfect Brain?

The absolute best possible score is about 0.7-0.8 (that's basically the randomness built into English — even a perfect brain can't predict truly creative or surprising writing). The best brain today scores 1.12. We're aiming for 0.9.

Getting there would be like going from a good student to a genius. It's possible, but it requires not just doing what everyone else does better — it requires doing things NOBODY has done before.

---

## 9. What We Learned (Discussion)

### 9.1 Small Boxes Force Smart Packing

When you only have a tiny box, you have to be really clever about what you put in it. Every number in the brain has to EARN its place. This is different from building a giant brain where you can be a bit wasteful. The tiny box makes you think harder, and that thinking leads to better designs.

### 9.2 Neat Rooms Work Better

We noticed something cool: the trick that makes the brain forget a tiny bit each step (weight decay) not only makes the brain smarter at guessing words, it also makes the brain EASIER TO PACK INTO THE SMALL BOX. It's like how a tidy room is both nicer to live in AND easier to photograph. Being organised helps in two ways at once.

### 9.3 Fair Tests Show Smarter Students

When we changed HOW we tested the brain (sliding window), the brain scored 0.03 points better WITHOUT getting any smarter. This means we were giving it an unfair test before — like testing kids by opening a book to a random page with no context. The brain was always smarter than we thought; we were just measuring it wrong.

### 9.4 You Don't Need a Giant Lab

We did all of this on a laptop in Cape Town. Yes, to actually compete we need to rent big computers for 10 minutes (costs about $4). But all the THINKING, INVENTING, and EXPERIMENTING happened right here. You don't need to be at Google or OpenAI to figure out how these things work. You just need curiosity and a good laptop.

---

## 10. The Dictionary Revolution (N-gram Caching)

### 10.1 What If the Brain Could Build Its Own Dictionary?

Imagine you're reading a book and you notice the author keeps using the same phrases over and over again. "Ladies and gentlemen" appears on every page. "The court hereby" shows up whenever someone speaks. After a while, you start to PREDICT those phrases before you even read them. You've built a little dictionary in your head of "things this author likes to say."

That's exactly what the n-gram cache does. While the brain is taking the test, it keeps a little notebook of word patterns it has already seen. Every time it sees a pair of words (like "of the") or a triple of words (like "in order to"), it writes it down. The next time those words appear, the brain checks its notebook and says "Oh! I've seen this pattern before — I bet I know what comes next!"

### 10.2 Mixing the Dictionary with the Brain

Here's the really clever bit: the brain doesn't ALWAYS listen to the dictionary. It has a mixing knob.

Think of it like this: when you're reading something really weird and confusing (high entropy), you listen MORE to your dictionary because the dictionary has simple, reliable patterns. When you're reading something predictable and you feel confident (low entropy), you trust your own brain more and turn the dictionary down.

This is called **entropy-adaptive mixing**. The brain measures how confused it is at each word. When confusion is HIGH, the dictionary gets turned up. When confusion is LOW, the brain's own guess takes priority. It's like having a friend who whispers answers to you — you only listen to them when you're stuck, and you trust yourself when you know the answer.

### 10.3 Why This Is So Cool

The best part? The dictionary doesn't take up any space in the tiny box! It gets built DURING the test, using patterns from the test itself. It's like being allowed to take notes during an exam — you don't need to bring any extra paper, you just use the margins of the exam itself.

The n-gram cache can improve the score by 0.01-0.02 BPB, which might sound small, but at this level of the competition, that's like shaving seconds off a Formula 1 lap time. Every fraction of a point matters.

---

## 11. Teaching the Brain During the Test (Test-Time Training)

### 11.1 Studying Each Question After You Answer It

Imagine you're taking a maths test, and after each question, your teacher lets you see the correct answer AND gives you five minutes to study the topic. By the time you get to question 10, you've basically had 10 mini-lessons tailored to exactly what's on this test. You'd do WAY better than if you just relied on what you studied before the exam.

That's what Test-Time Training (TTT) does. The brain takes the test, and after it scores each document, it gets to quickly LEARN from that document before moving on to the next one. The brain gets a little smarter with each document it reads.

### 11.2 The Score-First Protocol

There's an important rule: you have to score the document FIRST, then learn from it. You can't peek! That would be cheating. The protocol goes:

1. Read the document and make your predictions (this is the official score)
2. NOW learn from the document — adjust your brain slightly so you're better at this kind of text
3. Move on to the next document, where you'll be a tiny bit smarter

It's like a student who takes a quiz, gets the answers back, studies the corrections, and then takes the NEXT quiz a bit wiser. Perfectly fair, because each quiz was scored before any studying happened.

### 11.3 Resetting Between Documents

There's a tricky question: should the brain KEEP everything it learns from each document, or should it reset back to its original state between documents?

Turns out, resetting is often better. Why? Because what works great for a science article might mess up the brain's ability to read a cooking recipe. Each document is its own little world. So the brain learns from the document, uses that learning for similar text nearby, and then resets back to its clean starting state before tackling something completely different.

Think of it like stretching before a specific sport. You'd stretch differently for swimming vs running. You don't want your swimming stretches interfering with your running.

### 11.4 Why the Top Brain Uses This

The #1 brain in the entire competition uses TTT, and it's worth about 0.03 BPB — the same as our sliding window trick! That's a huge deal. It's basically free intelligence: the brain gets smarter DURING the test without needing any extra space in the tiny box.

---

## 12. Old School Meets New School (Classical-Neural Hybrid)

### 12.1 Before the Brain Was Born

Before neural networks existed, people still needed to compress text and predict words. They used good old-fashioned counting and pattern matching. Imagine keeping a giant tally chart: every time you see "the" followed by "cat," you add a tick mark. After reading a million sentences, you know that "the" is followed by "cat" about 2% of the time, "dog" about 1.5% of the time, "end" about 3% of the time, and so on.

These counting methods (called PPM, or "Prediction by Partial Matching") were the champions of text compression for DECADES before neural networks came along. They're simple, fast, and surprisingly good.

### 12.2 Combining the Old Tricks with the New Brain

Here's the insight: the old counting tricks and the new neural network brain are good at DIFFERENT things.

The counting tricks are brilliant at:
- Exact phrases that appear a lot ("United States of America")
- Short, common patterns ("of the", "in a", "to be")
- Patterns specific to the document you're reading right now

The neural network is brilliant at:
- Understanding meaning and context ("The president signed the ___" → probably "bill" or "agreement")
- Handling text it has never seen before
- Long-range patterns (remembering something from 500 words ago)

By combining them, you get the best of both worlds. The counting tricks handle the easy, repetitive patterns (and they're REALLY fast at it), which lets the neural network focus its brain power on the hard, creative, surprising parts.

### 12.3 How They Work Together

The combination works like a relay team:

1. **The counter** says: "Based on the last 2-3 words, here's my best guess at the next word, based purely on counting what I've seen before"
2. **The neural network** says: "Based on everything I understand about language and the full context, here's MY best guess"
3. **The mixer** combines both guesses: "Let's give 30% weight to the counter and 70% to the neural network — but if the counter is very confident about this particular pattern, bump it up to 50%"

This is the same entropy-adaptive mixing from the n-gram cache (Section 10), but applied more broadly. When the counting method has seen a pattern thousands of times and is very confident, it gets more say. When the pattern is rare or unusual, the neural network takes over.

### 12.4 Why This Matters for the Competition

The beautiful thing about classical methods is they use almost ZERO space in the tiny box. The counting tables get built during the test, just like the n-gram cache. So you're getting a significant boost in prediction quality without spending any of your precious 16 megabytes.

It's like discovering that your grandparents' old recipes are just as delicious as modern molecular gastronomy — and combining them makes a meal better than either one alone.

---

## 13. Where We're Going (Future Work)

### 13.1 The Realistic Target

If we combine everything we've built so far — the full neural network with all our tricks, plus n-gram caching, test-time training, and classical-neural hybrid methods — and run it all on the big competition computers for the full 10 minutes, we think we can hit **1.05-1.10 BPB**. That would put us in the top tier of the entire competition.

Here's the rough maths of how we get there:

| What | Expected improvement |
|------|---------------------|
| Our current architecture on big computers | ~1.15 BPB starting point |
| + Test-time training | -0.03 BPB |
| + N-gram caching | -0.01-0.02 BPB |
| + Classical-neural hybrid | -0.01 BPB |
| + Better quantisation (Soft-Round QAT) | -0.005 BPB |
| **Projected total** | **~1.05-1.10 BPB** |

### 13.2 The Moonshot

Could we break below 1.0 BPB? That would mean our tiny 16-megabyte brain can predict English text better than anything that's been done before at this size. It's possible, but it would need something nobody has invented yet — a fundamentally new way of thinking about the problem.

The absolute floor is about 0.7-0.8 BPB (the inherent randomness in English). Getting from 1.05 to 0.7 is like going from a really fast car to the speed of light — each step forward gets exponentially harder.

But that's what makes it fun. The people who break records are the ones who try things nobody else has thought of.

---

## 14. The End (Conclusion)

We took a computer brain that started as a random mess (scoring 2.41 — basically guessing) and turned it into something that genuinely understands English (scoring 1.47 and improving). We did it by stacking lots of clever tricks on top of each other, like building a tower of LEGO.

What we're most proud of:

1. **The Triple Dictionary**: We invented something nobody else tried — looking up three words at a time instead of two
2. **The Full Toolkit**: We've mapped out the path from basic neural network all the way to classical-neural hybrid systems, with n-gram caching and test-time training
3. **Writing Everything Down**: We showed exactly what each trick contributed, including the things that DIDN'T work
4. **Doing it from Cape Town**: Proving that smart AI research can happen anywhere
5. **The Toolbox**: We built a whole control panel (the dashboard) that makes it fun and easy to experiment

The biggest lesson? At the very edge of what's possible, the way you ORGANISE a small amount of intelligence matters more than just having MORE intelligence. Every brain cell has to count. Every number has to earn its place. And the cleverest tricks are the ones that make the brain smarter AND smaller at the same time.

The next lesson? Sometimes the best ideas come from looking BACKWARDS. The old counting methods from before neural networks were invented still have something to teach us. The future of AI might not just be about building bigger and fancier neural networks — it might be about combining them with the simple, reliable tricks that came before.

---

## The Back Pages (Appendices)

### What's Inside the Brain (Appendix B)

Think of the brain as a building with 11 floors:

**Ground Floor (Input):**
- A reception desk that converts words into number-codes (1,024 possible words -> 512 numbers each)
- A pair dictionary (10,240 entries for two-word combos)
- A triple dictionary (4,096 entries for three-word combos)
- A "peek at your neighbour" gadget

**Floors 1-11 (Thinking Floors):**
Each floor has two rooms:
- **The Looking Room (Attention)**: Where the brain looks around at all the other words and decides which ones matter. Has 8 little spotlights (attention heads) that each look for different things
- **The Thinking Room (MLP)**: Where the brain processes what it saw. Goes wide (512 -> 1536 numbers), thinks really hard, then squishes back down (1536 -> 512)

**Roof (Output):**
- Takes the final 512 numbers and converts them back into a guess about which of the 1,024 words comes next

There are also **elevators** (skip connections) connecting Floor 1 to Floor 11, Floor 2 to Floor 10, etc., so information from early thinking can jump directly to late thinking without getting lost.

**Total size**: 28,173,402 numbers. After squishing: fits in the 16-megabyte lunchbox.

### All the Knobs and Settings (Appendix C)

Like a recipe — all the ingredients and quantities, so someone else could make the exact same brain.

### What Tools We Used (Appendix D)

- MLX (Apple's brain-building kit for their own computers)
- A custom control panel we built ourselves (the dashboard at localhost:8888)
- zstd (a really good zip program)
- SentencePiece (the word-chopper that turns text into number-codes)

---

## Books We Read (References)

We didn't just make this up! We read papers by the people who invented these tricks, and we built on top of their ideas. Science is about standing on the shoulders of the people who came before you.

*(Same references as the grown-up version — even five-year-olds should know that good ideas come from building on other people's work.)*
