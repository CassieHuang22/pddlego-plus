(define (domain exploration-domain)
    (:requirements :strips :typing :negative-preconditions)
    (:types location direction)
    (:predicates
        (at ?l - location)
        (connected ?from - location ?to - location ?d - direction)
        (closed ?from - location ?to - location ?d - direction)
    )
    (:action move
        :parameters (?from - location ?to - location ?dir - direction)
        :precondition (and (at ?from) (connected ?from ?to ?dir) (not (closed ?from ?to ?dir)))
        :effect (and (not (at ?from)) (at ?to))
    )
    (:action open-door
        :parameters (?loc1 - location ?loc2 - location ?dir - direction)
        :precondition (and (at ?loc1) (connected ?loc1 ?loc2 ?dir) (closed ?loc1 ?loc2 ?dir))
        :effect (not (closed ?loc1 ?loc2 ?dir))
    )
)