import java.awt.*;

public class BlueBall extends Ball{

    public BlueBall(int xSpeed, int ySpeed, int ballSize) {
        super(Color.BLUE, xSpeed, ySpeed, ballSize);
    }

    @Override
    public void update(char keyChar) {
        this.setXSpeed(-1 * this.getXSpeed());
        this.setYSpeed(-1 * this.getYSpeed());
    }

    @Override
    public void update(Ball whiteBall) {
//        System.out.println("BlueBall update");
        if (this.isIntersect(whiteBall)){
            this.setXSpeed(-1 * this.getXSpeed());
            this.setYSpeed(-1 * this.getYSpeed());
        }
    }
}
