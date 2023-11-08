defmodule LearningElixir do
  require Nx.Defn

  import Nx.Defn

  @moduledoc """
  Documentation for `LearningElixir`.
  """

  @doc """
  Hello world.

  ## Examples

      iex> LearningElixir.hello()
      :world

  """
  def hello do
    :world
  end

  def tensor_operations do
    # Erstellt zwei Tensoren
    tensor1 = Nx.tensor([[1, 2], [3, 4]])
    tensor2 = Nx.tensor([[5, 6], [7, 8]])

    # Führt einige grundlegende Operationen aus
    sum = Nx.add(tensor1, tensor2)
    product = Nx.multiply(tensor1, tensor2)
    difference = Nx.subtract(tensor1, tensor2)

    {sum, product, difference}
  end

  def tensor_operations_2() do
    Nx.Defn.default_options(compiler: EXLA, client: :cuda)
    Nx.default_backend({EXLA.Backend, client: :cuda})
    # print backend name
    Nx.default_backend()

    # create a n * n tensor
    tensor1 = Nx.iota({10000, 10000}, type: {:f, 32})
    tensor2 = Nx.iota({10000, 10000}, type: {:f, 32})
    # rotate tensor2 by 180 degrees
    tensor2 = Nx.reverse(tensor2)
    # add 1 to each element of tensor2
    tensor2 = Nx.add(tensor2, 1)

    # basic operations
    sum = Nx.add(tensor1, tensor2)
    product = Nx.multiply(tensor1, tensor2)
    difference = Nx.subtract(tensor1, tensor2)
    quotient = Nx.divide(tensor1, tensor2)

    # basic operations 2
    sum_2 = Nx.add(sum, product)
    product_2 = Nx.multiply(product, difference)
    difference_2 = Nx.subtract(difference, quotient)
    quotient_2 = Nx.divide(quotient, sum)

    {sum_2, product_2, difference_2, quotient_2}
  end
end
