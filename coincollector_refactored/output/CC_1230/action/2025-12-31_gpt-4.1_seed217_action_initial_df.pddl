(define (domain house-exploration)
  (:requirements :strips :typing)
  (:types location direction)
  (:predicates 
    (at ?loc - location)
    (door ?from - location ?to - location ?dir - direction)
    (door-open ?from - location ?to - location ?dir - direction)
  )
  
  (:action open-door
    :parameters (?loc1 - location ?loc2 - location ?dir - direction)
    :precondition (and 
      (at ?loc1)
      (door ?loc1 ?loc2 ?dir)
      (not (door-open ?loc1 ?loc2 ?dir))
    )
    :effect (door-open ?loc1 ?loc2 ?dir)
  )

  (:action move
    :parameters (?from - location ?to - location ?dir - direction)
    :precondition (and 
      (at ?from)
      (door-open ?from ?to ?dir)
    )
    :effect (and 
      (not (at ?from))
      (at ?to)
    )
  )
)