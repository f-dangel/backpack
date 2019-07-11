Usage
====================================



.. code-block:: python
    :linenos:

    X, Y = get_data()
    model = make_model()
    loss = loss(model(X), Y)

    loss.backward()

    for p in model.parameters():
        print(p.grad)


.. code-block:: python
    :linenos:

    from backpack import bp, extend
    from backpack.extensions import Variance

    X, Y = get_data()
    model = make_model()
    extend(model)
    loss = loss(model(X), Y)

    with (backpack(Variance)):
        loss.backward()

        for p in model.parameters():
            print(p.grad)
            print(p.grad_variance)

