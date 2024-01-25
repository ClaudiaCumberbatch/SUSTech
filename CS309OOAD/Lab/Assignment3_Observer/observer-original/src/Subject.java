public interface Subject<T> {
    //when create a ball object, it is need to register the ball object into observer list.
    public void registerObserver(T t);
    // when restart a new game, it is need to remove all observer from paintinglist
    public void removeObserver(T t);
    // when clicked keyboard, it is need to notify all observers to update their state
    public void notifyObservers(char keyChar);
    // For task 2: when white ball moved, it is need to notify all observers to update their state
    public void notifyObservers();
    // or For task2 public void notifyObservers(int x, int y);
}