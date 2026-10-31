/**
 * Per-class content. Each class has:
 *  - memory: the fragment GLITCH gets back when the drawing is recognised;
 *  - echoes: two levels of hint, in words only. Echoes describe what makes the
 *    object recognisable; they never show a picture, because the game is about
 *    the player's own drawing, not tracing.
 * Classes without authored content (a retrained model with new labels) get
 * generic fallbacks built from the word itself.
 */

export interface ClassContent { memory: string; echoes: [string, string] }

export const CONTENT: Record<string, ClassContent> = {
  house: { memory: 'A front door left open on a summer evening. Somebody always drew the chimney smoking, even in July.',
    echoes: ['Shelter. Think of the outline a child draws first.', 'A box with a triangle on top. A door helps.'] },
  sun: { memory: 'Warmth on a metal casing. Every drawing of me outdoors had one of these in the corner, grinning.',
    echoes: ['The brightest thing in the sky, drawn simply.', 'A circle with short straight rays all the way around.'] },
  mountain: { memory: 'A horizon with teeth. People drew them in threes, the middle one tallest, snow on top.',
    echoes: ['Big, still, and pointy at the top.', 'One or more tall triangle peaks rising from a base line.'] },
  umbrella: { memory: 'Rain on fabric, a sound like static that is actually nice. Someone shared one with a stranger.',
    echoes: ['It keeps the rain off. It has a handle.', 'A wide dome on top, a straight stick down, a hook at the bottom.'] },
  star: { memory: 'Points of light I used to count when the Atlas was quiet. Five points. Always five.',
    echoes: ['Night sky. Pointy. Drawn in one continuous line by many people.', 'Five sharp points around a centre; the classic zig-zag outline.'] },
  apple: { memory: 'Red, round, a single leaf. One drawing had a bite taken out of it and I found that very funny.',
    echoes: ['Fruit. Round, with something sticking up at the top.', 'A round body with a small dip at the top, a short stem and a leaf.'] },
  book: { memory: 'Pages that smell like dust and glue. Someone fell asleep under one. The book did not mind.',
    echoes: ['Something you open to read.', 'A rectangle with a spine, or an open V shape with lines for pages.'] },
  clock: { memory: 'Tick. Tick. I used to set mine by the drawings: almost every one said ten past ten.',
    echoes: ['It tells you how late you are.', 'A circle with two hands from the centre; a few marks around the edge help.'] },
  chair: { memory: 'Four legs and patience. The chair by the window that nobody sat in, except the cat.',
    echoes: ['You sit on it. It has a back.', 'Seen from the side: a seat, a tall back, and legs reaching the floor.'] },
  cake: { memory: 'Candles counting upward. Frosting drawn as a wavy line. The best drawings had too many candles.',
    echoes: ['Birthdays. Layers. Something on top that you blow out.', 'A short wide cylinder or layers, with candles sticking up.'] },
  key: { memory: 'A small cold thing that opens a door. People drew them with round heads and toothy ends.',
    echoes: ['It opens a lock. Small, with teeth.', 'A ring or circle at one end, a long shaft, and notches at the other end.'] },
  banana: { memory: 'Yellow, curved, a little brown at the ends. My curve-detector loved bananas and I loved it back.',
    echoes: ['Fruit, curved, peelable.', 'A long, thick curved shape with a little stalk at one end. Thicker than a moon.'] },
  pizza: { memory: 'A triangle of warm, shared slowly. Little circles on top. Somebody always wanted the last slice.',
    echoes: ['Food, usually sliced. Toppings.', 'A triangle slice with a rounded crust edge, or a circle cut into wedges, with dots on top.'] },
  tree: { memory: 'Roots under the Atlas holding everything together. A cloud of leaves on a brown stick, mostly.',
    echoes: ['It grows. It has a trunk.', 'A straight trunk with a big rounded crown, or a stacked triangle like a pine.'] },
  flower: { memory: 'Petals in a ring. Someone drew one in every corner of a letter they never sent.',
    echoes: ['It grows in gardens. Petals.', 'A small centre circle, petals around it, a stem down with a leaf.'] },
  cloud: { memory: 'Soft edges drifting over the grid. The only shape in the Atlas that was allowed to change its mind.',
    echoes: ['Up in the sky, soft and puffy.', 'A wide outline made of several bumps, flatter along the bottom.'] },
  lightning: { memory: 'A bright crack across the sky, gone before you can draw it. People drew it anyway.',
    echoes: ['Storms. Very fast, very bright.', 'A bold zig-zag bolt, like a stretched Z, pointing downward.'] },
  fish: { memory: 'Silver flicker in the water layer. An oval, a triangle tail, one surprised eye.',
    echoes: ['It swims.', 'An oval body with a triangle tail at one end; an eye near the front.'] },
  butterfly: { memory: 'Two wings that match. People drew them more carefully than almost anything else.',
    echoes: ['It flutters. Matching halves.', 'A thin body down the middle, two big wings on each side, antennae on top.'] },
  moon: { memory: 'The night-light of the Atlas. Thin, curved, and calm. Never in a hurry.',
    echoes: ['Night sky, not a star.', 'A thin crescent: two curves meeting at sharp tips. Thinner than a banana, no stalk.'] },
  car: { memory: 'Engine hum, open windows, a road that goes somewhere better. Two wheels you can see, two you can’t.',
    echoes: ['It drives on roads.', 'A long body with a smaller cabin on top and two round wheels underneath.'] },
  bicycle: { memory: 'Two circles and a promise. The person who drew four hundred of these in a row, I salute you.',
    echoes: ['Two wheels, pedals, no engine.', 'Two equal circles side by side, joined by a triangle frame and handlebars.'] },
  airplane: { memory: 'A long white line across blue. Everyone waving at it, nobody sure if anyone waved back.',
    echoes: ['It flies, but it is a machine.', 'A long body, two wide wings across the middle, a small tail fin at the back.'] },
  sailboat: { memory: 'A triangle catching wind. The water was always drawn as two wobbly lines underneath.',
    echoes: ['It floats, pushed by wind.', 'A boat-shaped hull at the bottom and a tall triangle sail on a mast.'] },
  guitar: { memory: 'Six strings and a sound hole. Somebody played the same three chords until they were perfect.',
    echoes: ['An instrument you strum.', 'A rounded body like a figure eight, a hole in the middle, a long neck.'] },
  headphones: { memory: 'A private room you wear on your head. My favourite drawings had little music notes floating off them.',
    echoes: ['You wear it to listen.', 'A curved band arching over the top, with a round cup at each end.'] },
  cat: { memory: 'A warm weight that sits exactly where you are working. Triangle ears, whiskers, disapproval.',
    echoes: ['A small pet with opinions.', 'A round face with two pointed triangle ears on top and whiskers to the sides.'] },
  dog: { memory: 'Muddy paws, waggy line for a tail. The most loyal shape in the Atlas.',
    echoes: ['A pet that fetches.', 'A longer snout and floppy ears that hang down; a nose at the end.'] },
  bird: { memory: 'Wings in the morning. A little V shape in the distance meant a whole flock.',
    echoes: ['It flies and sings. Not a duck.', 'A small body, a pointed beak, and a wing or two; birds on a branch help.'] },
  duck: { memory: 'Quack-shaped. A body sitting on water, a flat bill, a slightly smug expression.',
    echoes: ['It floats on ponds. Not a bird in the sky.', 'A round body sitting on a water line, a head on top, and a wide flat bill.'] },
  frog: { memory: 'A small green philosopher on a lily pad. Big eyes, wide mouth, bent legs ready to go.',
    echoes: ['It hops. It croaks.', 'A wide flat body, two big eyes bulging on top, bent back legs.'] },
  rabbit: { memory: 'Two long ears that hear everything. A round puff of tail. Very fast, very soft.',
    echoes: ['It hops, with very long ears.', 'A round head with two long upright ears; a small round tail helps.'] },
};

export function contentFor(name: string): ClassContent {
  return CONTENT[name] ?? {
    memory: `The shape of ${name}. Not perfect, but unmistakably itself.`,
    echoes: [`Think about what makes ${name} different from everything else.`, `Draw the single feature a ${name} can’t be without, large and clear.`],
  };
}
