# **Dou Dizhu Game Flow & Structure**

---

## **1. Game Initialization**

* **Shuffle Deck**

  * Full deck = 54 cards (`0-12` x 4, `13` (joker), `14` (JOKER))
* **Take out reserved cards for landlord**

  * Last 3 cards after shuffling → reserved for the landlord
* **Deal cards**

  * Each player gets **17 cards** (sorted for convenience)

---

## **2. Decide the Landlord**

* Randomly choose a starting player to bid first
* **Bidding Process:**

  * Each player in turn chooses a bid (e.g., 1, 2, 3) or **pass**
  * Record bids in **GameInfo.risk**
  * Highest bidder becomes **landlord**
* **Landlord gets reserved cards**

  * Append 3 reserved cards to landlord’s hand and sort

---

## **3. Game State Representation**

* **GameInfo**

  * `risk`: stores bidding results → 3 rows, each `[player_index, bid_value]`
  * `moves`: history of all plays `[player, move_cards, move_type]`
* **Player**

  * `index`: 0, 1, 2
  * `cards`: numpy array of card ranks
  * `intelligence`: decision logic or AI
* **Reserved cards**: stored until landlord is chosen

---

### **Game Phase: Bidding for Landlord**

* Players bid in turn: 1, 2, 3 or pass
* Example:

  ```
  (A: 1)
  (B: 2)
  (C: pass)
  ```
* If **B wins**:

  * Landlord = B
  * Add reserved cards to B’s hand:

    ```
    reserved cards: [x, y, z]
    ```

---

## **4. Main Game Loop**

* Start from landlord
* On each turn:

  * Current player can:

    * Play a valid move (beats last move)
    * Or pass (if previous play exists)
  * Record move in **GameInfo.moves**
* Game ends when:

  * A player runs out of cards

---

## **5. Winning Conditions**

* If landlord finishes first → landlord wins
* If any peasant finishes first → both peasants win

---

## **6. Summary of Core Objects**

### **Game**

* `info` → `GameInfo`
* `p1`, `p2`, `p3` → `Player` objects
* `lord` → landlord index
* `reserved_cards` → last 3 cards of deck
* `playing_role` → whose turn now

### **GameInfo**

* `risk` → bidding history
* `moves` → move history

