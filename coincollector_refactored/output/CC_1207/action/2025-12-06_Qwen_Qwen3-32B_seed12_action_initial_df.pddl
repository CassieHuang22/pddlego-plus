(define (domain exploration)
  (:predicates
    (at ?loc - location)
    (closed ?loc1 - location ?dir - direction ?loc2 - location)
    (open ?loc1 - location ?dir - direction ?loc2 - location)
  )

  (:action open-door
    :parameters (?loc1 - location ?loc2 - location ?dir - direction)
    :precondition (and (at ?loc1) (closed ?loc1 ?dir ?loc2))
    :effect (and (not (closed ?loc1 ?dir ?loc2)) (open ?loc1 ?dir ?loc2))
  )

  (:action move
    :parameters (?from - location ?to - location ?dir - direction)
    :precondition (and (at ?from) (open ?from ?dir ?to))
    :effect (and (not (at ?from)) (at ?to))
  )
)
